/*
 * Copyright (C) 2015 Rob Clark <robclark@freedesktop.org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#include "util/debug.h"
#include "util/u_math.h"

#include "ir3_compiler.h"
#include "ir3_nir.h"
#include "ir3_shader.h"

static const nir_shader_compiler_options options = {
   .lower_fpow = true,
   .lower_scmp = true,
   .lower_flrp16 = true,
   .lower_flrp32 = true,
   .lower_flrp64 = true,
   .lower_ffract = true,
   .lower_fmod = true,
   .lower_fdiv = true,
   .lower_isign = true,
   .lower_ldexp = true,
   .lower_uadd_carry = true,
   .lower_usub_borrow = true,
   .lower_mul_high = true,
   .lower_mul_2x32_64 = true,
   .fuse_ffma16 = true,
   .fuse_ffma32 = true,
   .fuse_ffma64 = true,
   .vertex_id_zero_based = true,
   .lower_extract_byte = true,
   .lower_extract_word = true,
   .lower_insert_byte = true,
   .lower_insert_word = true,
   .lower_helper_invocation = true,
   .lower_bitfield_insert_to_shifts = true,
   .lower_bitfield_extract_to_shifts = true,
   .lower_pack_half_2x16 = true,
   .lower_pack_snorm_4x8 = true,
   .lower_pack_snorm_2x16 = true,
   .lower_pack_unorm_4x8 = true,
   .lower_pack_unorm_2x16 = true,
   .lower_unpack_half_2x16 = true,
   .lower_unpack_snorm_4x8 = true,
   .lower_unpack_snorm_2x16 = true,
   .lower_unpack_unorm_4x8 = true,
   .lower_unpack_unorm_2x16 = true,
   .lower_pack_split = true,
   .use_interpolated_input_intrinsics = true,
   .lower_rotate = true,
   .lower_to_scalar = true,
   .has_imul24 = true,
   .has_fsub = true,
   .has_isub = true,
   .lower_wpos_pntc = true,
   .lower_cs_local_index_from_id = true,

   /* Only needed for the spirv_to_nir() pass done in ir3_cmdline.c
    * but that should be harmless for GL since 64b is not
    * supported there.
    */
   .lower_int64_options = (nir_lower_int64_options)~0,
   .lower_uniforms_to_ubo = true,
   .use_scoped_barrier = true,
};

/* we don't want to lower vertex_id to _zero_based on newer gpus: */
static const nir_shader_compiler_options options_a6xx = {
   .lower_fpow = true,
   .lower_scmp = true,
   .lower_flrp16 = true,
   .lower_flrp32 = true,
   .lower_flrp64 = true,
   .lower_ffract = true,
   .lower_fmod = true,
   .lower_fdiv = true,
   .lower_isign = true,
   .lower_ldexp = true,
   .lower_uadd_carry = true,
   .lower_usub_borrow = true,
   .lower_mul_high = true,
   .lower_mul_2x32_64 = true,
   .fuse_ffma16 = true,
   .fuse_ffma32 = true,
   .fuse_ffma64 = true,
   .vertex_id_zero_based = false,
   .lower_extract_byte = true,
   .lower_extract_word = true,
   .lower_insert_byte = true,
   .lower_insert_word = true,
   .lower_helper_invocation = true,
   .lower_bitfield_insert_to_shifts = true,
   .lower_bitfield_extract_to_shifts = true,
   .lower_pack_half_2x16 = true,
   .lower_pack_snorm_4x8 = true,
   .lower_pack_snorm_2x16 = true,
   .lower_pack_unorm_4x8 = true,
   .lower_pack_unorm_2x16 = true,
   .lower_unpack_half_2x16 = true,
   .lower_unpack_snorm_4x8 = true,
   .lower_unpack_snorm_2x16 = true,
   .lower_unpack_unorm_4x8 = true,
   .lower_unpack_unorm_2x16 = true,
   .lower_pack_split = true,
   .use_interpolated_input_intrinsics = true,
   .lower_rotate = true,
   .vectorize_io = true,
   .lower_to_scalar = true,
   .has_imul24 = true,
   .has_fsub = true,
   .has_isub = true,
   .max_unroll_iterations = 32,
   .force_indirect_unrolling = nir_var_all,
   .lower_wpos_pntc = true,
   .lower_cs_local_index_from_id = true,

   /* Only needed for the spirv_to_nir() pass done in ir3_cmdline.c
    * but that should be harmless for GL since 64b is not
    * supported there.
    */
   .lower_int64_options = (nir_lower_int64_options)~0,
   .lower_uniforms_to_ubo = true,
   .lower_device_index_to_zero = true,
   .use_scoped_barrier = true,
};

const nir_shader_compiler_options *
ir3_get_compiler_options(struct ir3_compiler *compiler)
{
   if (compiler->gen >= 6)
      return &options_a6xx;
   return &options;
}

static bool
ir3_nir_should_vectorize_mem(unsigned align_mul, unsigned align_offset,
                             unsigned bit_size, unsigned num_components,
                             nir_intrinsic_instr *low,
                             nir_intrinsic_instr *high, void *data)
{
   assert(bit_size >= 8);
   if (bit_size != 32)
      return false;
   unsigned byte_size = bit_size / 8;

   int size = num_components * byte_size;

   /* Don't care about alignment past vec4. */
   assert(util_is_power_of_two_nonzero(align_mul));
   align_mul = MIN2(align_mul, 16);
   align_offset &= 15;

   /* Our offset alignment should aways be at least 4 bytes */
   if (align_mul < 4)
      return false;

   unsigned worst_start_offset = 16 - align_mul + align_offset;
   if (worst_start_offset + size > 16)
      return false;

   return true;
}

#define OPT(nir, pass, ...)                                                    \
   ({                                                                          \
      bool this_progress = false;                                              \
      NIR_PASS(this_progress, nir, pass, ##__VA_ARGS__);                       \
      this_progress;                                                           \
   })

#define OPT_V(nir, pass, ...) NIR_PASS_V(nir, pass, ##__VA_ARGS__)

void
ir3_optimize_loop(struct ir3_compiler *compiler, nir_shader *s)
{
   bool progress;
   unsigned lower_flrp = (s->options->lower_flrp16 ? 16 : 0) |
                         (s->options->lower_flrp32 ? 32 : 0) |
                         (s->options->lower_flrp64 ? 64 : 0);

   do {
      progress = false;

      OPT_V(s, nir_lower_vars_to_ssa);
      progress |= OPT(s, nir_opt_copy_prop_vars);
      progress |= OPT(s, nir_opt_dead_write_vars);
      progress |= OPT(s, nir_lower_alu_to_scalar, NULL, NULL);
      progress |= OPT(s, nir_lower_phis_to_scalar, false);

      progress |= OPT(s, nir_lower_undef_to_zero);
      progress |= OPT(s, nir_copy_prop);
      progress |= OPT(s, nir_opt_dce);
      progress |= OPT(s, nir_opt_cse);
      static int gcm = -1;
      if (gcm == -1)
         gcm = env_var_as_unsigned("GCM", 0);
      if (gcm == 1)
         progress |= OPT(s, nir_opt_gcm, true);
      else if (gcm == 2)
         progress |= OPT(s, nir_opt_gcm, false);
      progress |= OPT(s, nir_opt_peephole_select, 16, true, true);
      progress |= OPT(s, nir_opt_intrinsics);
      /* NOTE: GS lowering inserts an output var with varying slot that
       * is larger than VARYING_SLOT_MAX (ie. GS_VERTEX_FLAGS_IR3),
       * which triggers asserts in nir_shader_gather_info().  To work
       * around that skip lowering phi precision for GS.
       *
       * Calling nir_shader_gather_info() late also seems to cause
       * problems for tess lowering, for now since we only enable
       * fp16/int16 for frag and compute, skip phi precision lowering
       * for other stages.
       */
      if ((s->info.stage == MESA_SHADER_FRAGMENT) ||
          (s->info.stage == MESA_SHADER_COMPUTE) ||
          (s->info.stage == MESA_SHADER_KERNEL)) {
         progress |= OPT(s, nir_opt_phi_precision);
      }
      progress |= OPT(s, nir_opt_algebraic);
      progress |= OPT(s, nir_lower_alu);
      progress |= OPT(s, nir_lower_pack);
      progress |= OPT(s, nir_opt_constant_folding);

      nir_load_store_vectorize_options vectorize_opts = {
         .modes = nir_var_mem_ubo,
         .callback = ir3_nir_should_vectorize_mem,
         .robust_modes = compiler->robust_ubo_access ? nir_var_mem_ubo : 0,
      };
      progress |= OPT(s, nir_opt_load_store_vectorize, &vectorize_opts);

      if (lower_flrp != 0) {
         if (OPT(s, nir_lower_flrp, lower_flrp, false /* always_precise */)) {
            OPT(s, nir_opt_constant_folding);
            progress = true;
         }

         /* Nothing should rematerialize any flrps, so we only
          * need to do this lowering once.
          */
         lower_flrp = 0;
      }

      progress |= OPT(s, nir_opt_dead_cf);
      if (OPT(s, nir_opt_trivial_continues)) {
         progress |= true;
         /* If nir_opt_trivial_continues makes progress, then we need to clean
          * things up if we want any hope of nir_opt_if or nir_opt_loop_unroll
          * to make progress.
          */
         OPT(s, nir_copy_prop);
         OPT(s, nir_opt_dce);
      }
      progress |= OPT(s, nir_opt_if, false);
      progress |= OPT(s, nir_opt_loop_unroll);
      progress |= OPT(s, nir_opt_remove_phis);
      progress |= OPT(s, nir_opt_undef);
   } while (progress);
}

static bool
should_split_wrmask(const nir_instr *instr, const void *data)
{
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_store_shared:
   case nir_intrinsic_store_global:
   case nir_intrinsic_store_scratch:
      return true;
   default:
      return false;
   }
}

static bool
ir3_nir_lower_ssbo_size_filter(const nir_instr *instr, const void *data)
{
   return instr->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(instr)->intrinsic ==
             nir_intrinsic_get_ssbo_size;
}

static nir_ssa_def *
ir3_nir_lower_ssbo_size_instr(nir_builder *b, nir_instr *instr, void *data)
{
   uint8_t ssbo_size_to_bytes_shift = *(uint8_t *) data;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   return nir_ishl(b, &intr->dest.ssa, nir_imm_int(b, ssbo_size_to_bytes_shift));
}

/**
 * The resinfo opcode we have for getting the SSBO size on a6xx returns a byte
 * length divided by IBO_0_FMT, while the NIR intrinsic coming in is a number of
 * bytes. Switch things so the NIR intrinsic in our backend means dwords.
 */
static bool
ir3_nir_lower_ssbo_size(nir_shader *s, bool storage_16bit)
{
   uint8_t ssbo_size_to_bytes_shift = storage_16bit ? 1 : 2;
   return nir_shader_lower_instructions(s, ir3_nir_lower_ssbo_size_filter,
                                        ir3_nir_lower_ssbo_size_instr,
                                        &ssbo_size_to_bytes_shift);
}

void
ir3_nir_lower_io_to_temporaries(nir_shader *s)
{
   /* Outputs consumed by the VPC, VS inputs, and FS outputs are all handled
    * by the hardware pre-loading registers at the beginning and then reading
    * them at the end, so we can't access them indirectly except through
    * normal register-indirect accesses, and therefore ir3 doesn't support
    * indirect accesses on those. Other i/o is lowered in ir3_nir_lower_tess,
    * and indirects work just fine for those. GS outputs may be consumed by
    * VPC, but have their own lowering in ir3_nir_lower_gs() which does
    * something similar to nir_lower_io_to_temporaries so we shouldn't need
    * to lower them.
    *
    * Note: this might be a little inefficient for VS or TES outputs which are
    * when the next stage isn't an FS, but it probably don't make sense to
    * depend on the next stage before variant creation.
    *
    * TODO: for gallium, mesa/st also does some redundant lowering, including
    * running this pass for GS inputs/outputs which we don't want but not
    * including TES outputs or FS inputs which we do need. We should probably
    * stop doing that once we're sure all drivers are doing their own
    * indirect i/o lowering.
    */
   bool lower_input = s->info.stage == MESA_SHADER_VERTEX ||
                      s->info.stage == MESA_SHADER_FRAGMENT;
   bool lower_output = s->info.stage != MESA_SHADER_TESS_CTRL &&
                       s->info.stage != MESA_SHADER_GEOMETRY;
   if (lower_input || lower_output) {
      NIR_PASS_V(s, nir_lower_io_to_temporaries, nir_shader_get_entrypoint(s),
                 lower_output, lower_input);

      /* nir_lower_io_to_temporaries() creates global variables and copy
       * instructions which need to be cleaned up.
       */
      NIR_PASS_V(s, nir_split_var_copies);
      NIR_PASS_V(s, nir_lower_var_copies);
      NIR_PASS_V(s, nir_lower_global_vars_to_local);
   }

   /* Regardless of the above, we need to lower indirect references to
    * compact variables such as clip/cull distances because due to how
    * TCS<->TES IO works we cannot handle indirect accesses that "straddle"
    * vec4 components. nir_lower_indirect_derefs has a special case for
    * compact variables, so it will actually lower them even though we pass
    * in 0 modes.
    *
    * Using temporaries would be slightly better but
    * nir_lower_io_to_temporaries currently doesn't support TCS i/o.
    */
   NIR_PASS_V(s, nir_lower_indirect_derefs, 0, UINT32_MAX);
}

/*
 * Lowering for 64b intrinsics generated with OpenCL.  All our intrinsics
 * from a hw standpoint are 32b, so we just need to combine in zero for
 * the upper 32bits and let the other nir passes clean up the mess.
 */

static bool
is_intrinsic_store(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_store_global:
   case nir_intrinsic_store_shared:
   case nir_intrinsic_store_scratch:
      return true;
   default:
      return false;
   }
}

static bool
is_intrinsic_load(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_constant:
   case nir_intrinsic_load_shared:
   case nir_intrinsic_load_scratch:
      return true;
   default:
      return false;
   }
}

static bool
lower_64b_intrinsics_filter(const nir_instr *instr, const void *unused)
{
   (void)unused;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   if (is_intrinsic_store(intr->intrinsic))
      return nir_src_bit_size(intr->src[0]) == 64;

   if (nir_intrinsic_dest_components(intr) == 0)
      return false;

   return nir_dest_bit_size(intr->dest) == 64;
}

static nir_ssa_def *
lower_64b_intrinsics(nir_builder *b, nir_instr *instr, void *unused)
{
   (void)unused;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   /* We could be *slightly* more clever and, for ex, turn a 64b vec4
    * load into two 32b vec4 loads, rather than 4 32b vec2 loads.
    */

   if (is_intrinsic_store(intr->intrinsic)) {
      unsigned num_comp = nir_intrinsic_src_components(intr, 0);
      unsigned wrmask = nir_intrinsic_write_mask(intr);
      nir_ssa_def *val = nir_ssa_for_src(b, intr->src[0], num_comp);
      nir_ssa_def *off = nir_ssa_for_src(b, intr->src[1], 1);

      for (unsigned i = 0; i < num_comp; i++) {
         if (!(wrmask & BITFIELD_BIT(i)))
            continue;

         nir_ssa_def *c64 = nir_channel(b, val, i);
         nir_ssa_def *c32 = nir_unpack_64_2x32(b, c64);

         nir_intrinsic_instr *store =
               nir_intrinsic_instr_create(b->shader, intr->intrinsic);
         store->num_components = 2;
         store->src[0] = nir_src_for_ssa(c32);
         store->src[1] = nir_src_for_ssa(off);
         nir_intrinsic_set_align(store, nir_intrinsic_align(intr), 0);
         nir_intrinsic_set_write_mask(store, 0x3);
         nir_builder_instr_insert(b, &store->instr);

         off = nir_iadd(b, off, nir_imm_intN_t(b, 8, off->bit_size));
      }

      return NIR_LOWER_INSTR_PROGRESS_REPLACE;
   }

   unsigned num_comp = nir_intrinsic_dest_components(intr);

   nir_ssa_def *def = &intr->dest.ssa;
   def->bit_size = 32;

   /* load_kernel_input is handled specially, lowering to two 32b inputs:
    */
   if (intr->intrinsic == nir_intrinsic_load_kernel_input) {
      assert(num_comp == 1);

      nir_ssa_def *offset = nir_iadd(b,
            nir_ssa_for_src(b, intr->src[0], 1),
            nir_imm_int(b, 4));

      nir_ssa_def *upper = nir_build_load_kernel_input(
            b, 1, 32, offset);

      return nir_pack_64_2x32_split(b, def, upper);
   }

   nir_ssa_def *components[num_comp];

   if (is_intrinsic_load(intr->intrinsic)) {
      nir_ssa_def *off = nir_ssa_for_src(b, intr->src[0], 1);

      for (unsigned i = 0; i < num_comp; i++) {
         nir_intrinsic_instr *load =
            nir_intrinsic_instr_create(b->shader, intr->intrinsic);
         load->num_components = 2;
         load->src[0] = nir_src_for_ssa(off);
         nir_intrinsic_set_align(load, nir_intrinsic_align(intr), 0);
         nir_ssa_dest_init(&load->instr, &load->dest, 2, 32, NULL);
         nir_builder_instr_insert(b, &load->instr);

         components[i] = nir_pack_64_2x32(b, &load->dest.ssa);

         off = nir_iadd(b, off, nir_imm_intN_t(b, 8, off->bit_size));
      }
   } else {
      /* The remaining (non load/store) intrinsics just get zero-
       * extended from 32b to 64b:
       */
      for (unsigned i = 0; i < num_comp; i++) {
         nir_ssa_def *c = nir_channel(b, def, i);
         components[i] = nir_pack_64_2x32_split(b, c, nir_imm_zero(b, 1, 32));
      }
   }

   return nir_build_alu_src_arr(b, nir_op_vec(num_comp), components);
}

static bool
ir3_nir_lower_64b_intrinsics(nir_shader *shader)
{
   return nir_shader_lower_instructions(
         shader, lower_64b_intrinsics_filter,
         lower_64b_intrinsics, NULL);
}

/*
 * Lowering for wide (larger than vec4) load/store
 */

static bool
lower_wide_load_store_filter(const nir_instr *instr, const void *unused)
{
   (void)unused;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   if (is_intrinsic_store(intr->intrinsic))
      return nir_intrinsic_src_components(intr, 0) > 4;

   if (is_intrinsic_load(intr->intrinsic))
      return nir_intrinsic_dest_components(intr) > 4;

   return false;
}

static nir_ssa_def *
lower_wide_load_store(nir_builder *b, nir_instr *instr, void *unused)
{
   (void)unused;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   if (is_intrinsic_store(intr->intrinsic)) {
      unsigned num_comp = nir_intrinsic_src_components(intr, 0);
      unsigned wrmask = nir_intrinsic_write_mask(intr);
      nir_ssa_def *val = nir_ssa_for_src(b, intr->src[0], num_comp);
      nir_ssa_def *addr = nir_ssa_for_src(b, intr->src[1], 1);

      for (unsigned off = 0; off < num_comp; off += 4) {
         unsigned c = MIN2(num_comp - off, 4);
         nir_ssa_def *v = nir_channels(b, val, BITFIELD_MASK(c) << off);

         nir_intrinsic_instr *store =
               nir_intrinsic_instr_create(b->shader, intr->intrinsic);
         store->num_components = c;
         store->src[0] = nir_src_for_ssa(v);
         store->src[1] = nir_src_for_ssa(addr);
         nir_intrinsic_set_align(store, nir_intrinsic_align(intr), 0);
         nir_intrinsic_set_write_mask(store, (wrmask >> off) & 0xf);
         nir_builder_instr_insert(b, &store->instr);

         addr = nir_iadd(b,
               nir_imm_intN_t(b, (c * val->bit_size) / 8, addr->bit_size),
               addr);
      }

      return NIR_LOWER_INSTR_PROGRESS_REPLACE;
   } else {
      unsigned num_comp = nir_intrinsic_dest_components(intr);
      unsigned bit_size = nir_dest_bit_size(intr->dest);
      nir_ssa_def *addr = nir_ssa_for_src(b, intr->src[0], 1);
      nir_ssa_def *components[num_comp];

      for (unsigned off = 0; off < num_comp;) {
         unsigned c = MIN2(num_comp - off, 4);

         nir_intrinsic_instr *load =
            nir_intrinsic_instr_create(b->shader, intr->intrinsic);
         load->num_components = c;
         load->src[0] = nir_src_for_ssa(addr);
         nir_intrinsic_set_align(load, nir_intrinsic_align(intr), 0);
         nir_ssa_dest_init(&load->instr, &load->dest, c, bit_size, NULL);
         nir_builder_instr_insert(b, &load->instr);

         addr = nir_iadd(b,
               nir_imm_intN_t(b, (c * bit_size) / 8, addr->bit_size),
               addr);

         for (unsigned i = 0; i < c; i++) {
            components[off++] = nir_channel(b, &load->dest.ssa, i);
         }
      }

      return nir_build_alu_src_arr(b, nir_op_vec(num_comp), components);
   }
}

static bool
ir3_nir_lower_wide_load_store(nir_shader *shader)
{
   return nir_shader_lower_instructions(
         shader, lower_wide_load_store_filter,
         lower_wide_load_store, NULL);
}

/*
 * Lowering for load_global/store_global to ir3 variants:
 */

static bool
lower_load_store_global_filter(const nir_instr *instr, const void *unused)
{
   (void)unused;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   return (intr->intrinsic == nir_intrinsic_load_global) ||
          (intr->intrinsic == nir_intrinsic_load_global_constant) ||
          (intr->intrinsic == nir_intrinsic_store_global);
}

static nir_ssa_def *
lower_load_store_global(nir_builder *b, nir_instr *instr, void *unused)
{
   (void)unused;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   bool load = intr->intrinsic != nir_intrinsic_store_global;

   nir_ssa_def *addr64 = nir_ssa_for_src(b, intr->src[load ? 0 : 1], 1);
   nir_ssa_def *addr = nir_unpack_64_2x32(b, addr64);

   /*
    * Note that we can get vec8/vec16 with OpenCL.. we need to split
    * those up into max 4 components per load/store.
    *
    * TODO move wide vec handling into the pass that lowers wide
    * scratch/shared.. and add a helper that the nir->ir3 code could
    * use to fish out constant offsets and encode them into the
    * instruction to avoid extra alu instructions??
    */

   if (load) {
      unsigned num_comp = nir_intrinsic_dest_components(intr);
      nir_ssa_def *components[num_comp];
      for (unsigned off = 0; off < num_comp;) {
         unsigned c = MIN2(num_comp - off, 4);
         nir_ssa_def *val = nir_build_load_global_ir3(
               b, c, nir_dest_bit_size(intr->dest),
               addr, nir_imm_int(b, off));
         for (unsigned i = 0; i < c; i++) {
            components[off++] = nir_channel(b, val, i);
         }
      }
      return nir_build_alu_src_arr(b, nir_op_vec(num_comp), components);
   } else {
      unsigned num_comp = nir_intrinsic_src_components(intr, 0);
      nir_ssa_def *value = nir_ssa_for_src(b, intr->src[0], num_comp);
      for (unsigned off = 0; off < num_comp; off += 4) {
         unsigned c = MIN2(num_comp - off, 4);
         nir_ssa_def *v = nir_channels(b, value, BITFIELD_MASK(c) << off);
         nir_build_store_global_ir3(b, v, addr, nir_imm_int(b, off));
      }
      return NIR_LOWER_INSTR_PROGRESS_REPLACE;
   }
}

static bool
ir3_nir_lower_load_store_global(nir_shader *shader)
{
   return nir_shader_lower_instructions(
         shader, lower_load_store_global_filter,
         lower_load_store_global, NULL);
}

/*
 * Pass to lower 64b phi to vec2_32
 */

static bool
lower_64b_phi(nir_builder *b, nir_phi_instr *phi)
{
   assert(phi->dest.is_ssa);
   assert(phi->dest.ssa.num_components == 1);

   if (phi->dest.ssa.bit_size != 64)
      return false;

   nir_phi_instr *new_phi = nir_phi_instr_create(b->shader);
   nir_ssa_dest_init(&new_phi->instr, &new_phi->dest, 2, 32, NULL);

   /* Push the conversion to vec2_32 into the new phi sources: */
   nir_foreach_phi_src (src, phi) {
      assert(src->src.is_ssa);

      /* insert conversion to vec2_32 in block of original phi src: */
      b->cursor = nir_after_instr_and_phis(src->src.ssa->parent_instr);
      nir_ssa_def *old_src = src->src.ssa;
      nir_ssa_def *new_src = nir_unpack_64_2x32(b, old_src);

      /* and add corresponding phi_src to the new_phi: */
      nir_phi_instr_add_src(new_phi, src->pred, nir_src_for_ssa(new_src));
   }

   /* Then insert the new phi after all sources are in place: */
   b->cursor = nir_after_instr(&phi->instr);
   nir_builder_instr_insert(b, &new_phi->instr);

   /* And finally add conversion back to 64b after the phi, and re-write
    * the original phi's uses.  If the phi's consumers have been lowered
    * from 64b, then the other opt passes should clean up the conversion
    * back, leaving only 32b instructions remaining.
    */
   b->cursor = nir_after_instr_and_phis(&new_phi->instr);
   nir_ssa_def *def = nir_pack_64_2x32(b, &new_phi->dest.ssa);

   nir_ssa_def_rewrite_uses(&phi->dest.ssa, def);

   return true;
}

static bool
ir3_nir_lower_64b_phi(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function (function, shader) {
      if (!function->impl)
         continue;

      nir_builder b;
      nir_builder_init(&b, function->impl);

      nir_foreach_block (block, function->impl) {
         nir_foreach_instr_safe (instr, block) {
            if (instr->type != nir_instr_type_phi)
               break;

            progress |= lower_64b_phi(&b, nir_instr_as_phi(instr));
         }
      }

      if (progress) {
         nir_metadata_preserve(function->impl,
                               nir_metadata_block_index |
                               nir_metadata_dominance);
      } else {
         nir_metadata_preserve(function->impl, nir_metadata_all);
      }
   }

   return progress;
}

void
ir3_finalize_nir(struct ir3_compiler *compiler, nir_shader *s)
{
   struct nir_lower_tex_options tex_options = {
      .lower_rect = 0,
      .lower_tg4_offsets = true,
   };

   if (compiler->gen >= 4) {
      /* a4xx seems to have *no* sam.p */
      tex_options.lower_txp = ~0; /* lower all txp */
   } else {
      /* a3xx just needs to avoid sam.p for 3d tex */
      tex_options.lower_txp = (1 << GLSL_SAMPLER_DIM_3D);
   }

   if (ir3_shader_debug & IR3_DBG_DISASM) {
      mesa_logi("----------------------");
      nir_log_shaderi(s);
      mesa_logi("----------------------");
   }

   if (s->info.stage == MESA_SHADER_GEOMETRY)
      NIR_PASS_V(s, ir3_nir_lower_gs);

   NIR_PASS_V(s, nir_lower_amul, ir3_glsl_type_size);

   OPT_V(s, nir_lower_regs_to_ssa);
   OPT_V(s, nir_lower_wrmasks, should_split_wrmask, s);

   OPT_V(s, nir_lower_tex, &tex_options);
   OPT_V(s, nir_lower_load_const_to_scalar);
   if (compiler->gen < 5)
      OPT_V(s, ir3_nir_lower_tg4_to_tex);

   /* Note bitwise OR intentional: */
   if (OPT(s, ir3_nir_lower_64b_intrinsics) |
       OPT(s, ir3_nir_lower_wide_load_store))
      OPT_V(s, nir_lower_int64);

   OPT_V(s, ir3_nir_lower_load_store_global);

   if (OPT(s, ir3_nir_lower_64b_phi)) {
      ; // probably need to scalarize phis...
   }

   ir3_optimize_loop(compiler, s);

   /* do idiv lowering after first opt loop to get a chance to propagate
    * constants for divide by immed power-of-two:
    */
   nir_lower_idiv_options idiv_options = {
      .imprecise_32bit_lowering = true,
      .allow_fp16 = true,
   };
   const bool idiv_progress = OPT(s, nir_lower_idiv, &idiv_options);

   if (idiv_progress)
      ir3_optimize_loop(compiler, s);

   OPT_V(s, nir_remove_dead_variables, nir_var_function_temp, NULL);

   if (ir3_shader_debug & IR3_DBG_DISASM) {
      mesa_logi("----------------------");
      nir_log_shaderi(s);
      mesa_logi("----------------------");
   }

   /* st_program.c's parameter list optimization requires that future nir
    * variants don't reallocate the uniform storage, so we have to remove
    * uniforms that occupy storage.  But we don't want to remove samplers,
    * because they're needed for YUV variant lowering.
    */
   nir_foreach_uniform_variable_safe (var, s) {
      if (var->data.mode == nir_var_uniform &&
          (glsl_type_get_image_count(var->type) ||
           glsl_type_get_sampler_count(var->type)))
         continue;

      exec_node_remove(&var->node);
   }
   nir_validate_shader(s, "after uniform var removal");

   nir_sweep(s);
}

static bool
lower_subgroup_id_filter(const nir_instr *instr, const void *unused)
{
   (void)unused;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   return intr->intrinsic == nir_intrinsic_load_subgroup_invocation ||
          intr->intrinsic == nir_intrinsic_load_subgroup_id ||
          intr->intrinsic == nir_intrinsic_load_num_subgroups;
}

static nir_ssa_def *
lower_subgroup_id(nir_builder *b, nir_instr *instr, void *unused)
{
   (void)unused;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic == nir_intrinsic_load_subgroup_invocation) {
      return nir_iand(
         b, nir_load_local_invocation_index(b),
         nir_isub(b, nir_load_subgroup_size(b), nir_imm_int(b, 1)));
   } else if (intr->intrinsic == nir_intrinsic_load_subgroup_id) {
      return nir_ishr(b, nir_load_local_invocation_index(b),
                      nir_load_subgroup_id_shift_ir3(b));
   } else {
      assert(intr->intrinsic == nir_intrinsic_load_num_subgroups);
      /* If the workgroup size is constant,
       * nir_lower_compute_system_values() will replace local_size with a
       * constant so this can mostly be constant folded away.
       */
      nir_ssa_def *local_size = nir_load_workgroup_size(b);
      nir_ssa_def *size =
         nir_imul24(b, nir_channel(b, local_size, 0),
                    nir_imul24(b, nir_channel(b, local_size, 1),
                               nir_channel(b, local_size, 2)));
      nir_ssa_def *one = nir_imm_int(b, 1);
      return nir_iadd(b, one,
                      nir_ishr(b, nir_isub(b, size, one),
                               nir_load_subgroup_id_shift_ir3(b)));
   }
}

static bool
ir3_nir_lower_subgroup_id_cs(nir_shader *shader)
{
   return nir_shader_lower_instructions(shader, lower_subgroup_id_filter,
                                        lower_subgroup_id, NULL);
}

/**
 * Late passes that need to be done after pscreen->finalize_nir()
 */
void
ir3_nir_post_finalize(struct ir3_compiler *compiler, nir_shader *s)
{
   NIR_PASS_V(s, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
              ir3_glsl_type_size, (nir_lower_io_options)0);

   if (s->info.stage == MESA_SHADER_FRAGMENT) {
      /* NOTE: lower load_barycentric_at_sample first, since it
       * produces load_barycentric_at_offset:
       */
      NIR_PASS_V(s, ir3_nir_lower_load_barycentric_at_sample);
      NIR_PASS_V(s, ir3_nir_lower_load_barycentric_at_offset);
      NIR_PASS_V(s, ir3_nir_move_varying_inputs);
      NIR_PASS_V(s, nir_lower_fb_read);
   }

   if (compiler->gen >= 6 && s->info.stage == MESA_SHADER_FRAGMENT &&
       !(ir3_shader_debug & IR3_DBG_NOFP16)) {
      NIR_PASS_V(s, nir_lower_mediump_io, nir_var_shader_out, 0, false);
   }

   if ((s->info.stage == MESA_SHADER_COMPUTE) ||
       (s->info.stage == MESA_SHADER_KERNEL)) {
      bool progress = false;
      NIR_PASS(progress, s, nir_lower_subgroups,
               &(nir_lower_subgroups_options){
                  .subgroup_size = 128,
                  .ballot_bit_size = 32,
                  .ballot_components = 4,
                  .lower_to_scalar = true,
                  .lower_vote_eq = true,
                  .lower_subgroup_masks = true,
                  .lower_read_invocation_to_cond = true,
               });

      progress = false;
      NIR_PASS(progress, s, ir3_nir_lower_subgroup_id_cs);

      /* ir3_nir_lower_subgroup_id_cs creates extra compute intrinsics which
       * we need to lower again.
       */
      if (progress)
         NIR_PASS_V(s, nir_lower_compute_system_values, NULL);
   }

   /* we cannot ensure that ir3_finalize_nir() is only called once, so
    * we also need to do any run-once workarounds here:
    */
   OPT_V(s, ir3_nir_apply_trig_workarounds);

   const nir_lower_image_options lower_image_opts = {
      .lower_cube_size = true,
   };
   NIR_PASS_V(s, nir_lower_image, &lower_image_opts);

   const nir_lower_idiv_options lower_idiv_options = {
      .imprecise_32bit_lowering = true,
      .allow_fp16 = true,
   };
   NIR_PASS_V(s, nir_lower_idiv, &lower_idiv_options); /* idiv generated by cube lowering */

   if (compiler->gen >= 6)
      OPT_V(s, ir3_nir_lower_ssbo_size, compiler->storage_16bit);

   ir3_optimize_loop(compiler, s);
}

static bool
ir3_nir_lower_view_layer_id(nir_shader *nir, bool layer_zero, bool view_zero)
{
   unsigned layer_id_loc = ~0, view_id_loc = ~0;
   nir_foreach_shader_in_variable (var, nir) {
      if (var->data.location == VARYING_SLOT_LAYER)
         layer_id_loc = var->data.driver_location;
      if (var->data.location == VARYING_SLOT_VIEWPORT)
         view_id_loc = var->data.driver_location;
   }

   assert(!layer_zero || layer_id_loc != ~0);
   assert(!view_zero || view_id_loc != ~0);

   bool progress = false;
   nir_builder b;

   nir_foreach_function (func, nir) {
      nir_builder_init(&b, func->impl);

      nir_foreach_block (block, func->impl) {
         nir_foreach_instr_safe (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

            if (intrin->intrinsic != nir_intrinsic_load_input)
               continue;

            unsigned base = nir_intrinsic_base(intrin);
            if (base != layer_id_loc && base != view_id_loc)
               continue;

            b.cursor = nir_before_instr(&intrin->instr);
            nir_ssa_def *zero = nir_imm_int(&b, 0);
            nir_ssa_def_rewrite_uses(&intrin->dest.ssa, zero);
            nir_instr_remove(&intrin->instr);
            progress = true;
         }
      }

      if (progress) {
         nir_metadata_preserve(
            func->impl, nir_metadata_block_index | nir_metadata_dominance);
      } else {
         nir_metadata_preserve(func->impl, nir_metadata_all);
      }
   }

   return progress;
}

void
ir3_nir_lower_variant(struct ir3_shader_variant *so, nir_shader *s)
{
   if (ir3_shader_debug & IR3_DBG_DISASM) {
      mesa_logi("----------------------");
      nir_log_shaderi(s);
      mesa_logi("----------------------");
   }

   bool progress = false;

   if (so->key.has_gs || so->key.tessellation) {
      switch (so->shader->type) {
      case MESA_SHADER_VERTEX:
         NIR_PASS_V(s, ir3_nir_lower_to_explicit_output, so,
                    so->key.tessellation);
         progress = true;
         break;
      case MESA_SHADER_TESS_CTRL:
         NIR_PASS_V(s, ir3_nir_lower_tess_ctrl, so, so->key.tessellation);
         NIR_PASS_V(s, ir3_nir_lower_to_explicit_input, so);
         progress = true;
         break;
      case MESA_SHADER_TESS_EVAL:
         NIR_PASS_V(s, ir3_nir_lower_tess_eval, so, so->key.tessellation);
         if (so->key.has_gs)
            NIR_PASS_V(s, ir3_nir_lower_to_explicit_output, so,
                       so->key.tessellation);
         progress = true;
         break;
      case MESA_SHADER_GEOMETRY:
         NIR_PASS_V(s, ir3_nir_lower_to_explicit_input, so);
         progress = true;
         break;
      default:
         break;
      }
   }

   if (s->info.stage == MESA_SHADER_VERTEX) {
      if (so->key.ucp_enables)
         progress |=
            OPT(s, nir_lower_clip_vs, so->key.ucp_enables, false, false, NULL);
   } else if (s->info.stage == MESA_SHADER_FRAGMENT) {
      bool layer_zero =
         so->key.layer_zero && (s->info.inputs_read & VARYING_BIT_LAYER);
      bool view_zero =
         so->key.view_zero && (s->info.inputs_read & VARYING_BIT_VIEWPORT);

      if (so->key.ucp_enables && !so->shader->compiler->has_clip_cull)
         progress |= OPT(s, nir_lower_clip_fs, so->key.ucp_enables, false);
      if (layer_zero || view_zero)
         progress |= OPT(s, ir3_nir_lower_view_layer_id, layer_zero, view_zero);
   }

   /* Move large constant variables to the constants attached to the NIR
    * shader, which we will upload in the immediates range.  This generates
    * amuls, so we need to clean those up after.
    *
    * Passing no size_align, we would get packed values, which if we end up
    * having to load with LDC would result in extra reads to unpack from
    * straddling loads.  Align everything to vec4 to avoid that, though we
    * could theoretically do better.
    */
   OPT_V(s, nir_opt_large_constants, glsl_get_vec4_size_align_bytes,
         32 /* bytes */);
   OPT_V(s, ir3_nir_lower_load_constant, so);

   if (!so->binning_pass)
      OPT_V(s, ir3_nir_analyze_ubo_ranges, so);

   progress |= OPT(s, ir3_nir_lower_ubo_loads, so);

   /* Lower large temporaries to scratch, which in Qualcomm terms is private
    * memory, to avoid excess register pressure. This should happen after
    * nir_opt_large_constants, because loading from a UBO is much, much less
    * expensive.
    */
   if (so->shader->compiler->has_pvtmem) {
      progress |= OPT(s, nir_lower_vars_to_scratch, nir_var_function_temp,
                      16 * 16 /* bytes */, glsl_get_natural_size_align_bytes);
   }

   /* Lower scratch writemasks */
   progress |= OPT(s, nir_lower_wrmasks, should_split_wrmask, s);

   OPT_V(s, nir_lower_amul, ir3_glsl_type_size);

   /* UBO offset lowering has to come after we've decided what will
    * be left as load_ubo
    */
   if (so->shader->compiler->gen >= 6)
      progress |= OPT(s, nir_lower_ubo_vec4);

   OPT_V(s, ir3_nir_lower_io_offsets);

   if (progress)
      ir3_optimize_loop(so->shader->compiler, s);

   /* Fixup indirect load_uniform's which end up with a const base offset
    * which is too large to encode.  Do this late(ish) so we actually
    * can differentiate indirect vs non-indirect.
    */
   if (OPT(s, ir3_nir_fixup_load_uniform))
      ir3_optimize_loop(so->shader->compiler, s);

   /* Do late algebraic optimization to turn add(a, neg(b)) back into
    * subs, then the mandatory cleanup after algebraic.  Note that it may
    * produce fnegs, and if so then we need to keep running to squash
    * fneg(fneg(a)).
    */
   bool more_late_algebraic = true;
   while (more_late_algebraic) {
      more_late_algebraic = OPT(s, nir_opt_algebraic_late);
      OPT_V(s, nir_opt_constant_folding);
      OPT_V(s, nir_copy_prop);
      OPT_V(s, nir_opt_dce);
      OPT_V(s, nir_opt_cse);
   }

   OPT_V(s, nir_opt_sink, nir_move_const_undef);

   if (ir3_shader_debug & IR3_DBG_DISASM) {
      mesa_logi("----------------------");
      nir_log_shaderi(s);
      mesa_logi("----------------------");
   }

   nir_sweep(s);

   /* Binning pass variants re-use  the const_state of the corresponding
    * draw pass shader, so that same const emit can be re-used for both
    * passes:
    */
   if (!so->binning_pass)
      ir3_setup_const_state(s, so, ir3_const_state(so));
}

static void
ir3_nir_scan_driver_consts(struct ir3_compiler *compiler, nir_shader *shader, struct ir3_const_state *layout)
{
   nir_foreach_function (function, shader) {
      if (!function->impl)
         continue;

      nir_foreach_block (block, function->impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            unsigned idx;

            switch (intr->intrinsic) {
            case nir_intrinsic_image_atomic_add:
            case nir_intrinsic_image_atomic_imin:
            case nir_intrinsic_image_atomic_umin:
            case nir_intrinsic_image_atomic_imax:
            case nir_intrinsic_image_atomic_umax:
            case nir_intrinsic_image_atomic_and:
            case nir_intrinsic_image_atomic_or:
            case nir_intrinsic_image_atomic_xor:
            case nir_intrinsic_image_atomic_exchange:
            case nir_intrinsic_image_atomic_comp_swap:
            case nir_intrinsic_image_load:
            case nir_intrinsic_image_store:
            case nir_intrinsic_image_size:
               if (compiler->gen < 6 &&
                   !(intr->intrinsic == nir_intrinsic_image_load &&
                     !(nir_intrinsic_access(intr) & ACCESS_COHERENT))) {
                  idx = nir_src_as_uint(intr->src[0]);
                  if (layout->image_dims.mask & (1 << idx))
                     break;
                  layout->image_dims.mask |= (1 << idx);
                  layout->image_dims.off[idx] = layout->image_dims.count;
                  layout->image_dims.count += 3; /* three const per */
               }
               break;
            case nir_intrinsic_load_base_vertex:
            case nir_intrinsic_load_first_vertex:
               layout->num_driver_params =
                  MAX2(layout->num_driver_params, IR3_DP_VTXID_BASE + 1);
               break;
            case nir_intrinsic_load_base_instance:
               layout->num_driver_params =
                  MAX2(layout->num_driver_params, IR3_DP_INSTID_BASE + 1);
               break;
            case nir_intrinsic_load_user_clip_plane:
               idx = nir_intrinsic_ucp_id(intr);
               layout->num_driver_params = MAX2(layout->num_driver_params,
                                                IR3_DP_UCP0_X + (idx + 1) * 4);
               break;
            case nir_intrinsic_load_num_workgroups:
               layout->num_driver_params =
                  MAX2(layout->num_driver_params, IR3_DP_NUM_WORK_GROUPS_Z + 1);
               break;
            case nir_intrinsic_load_workgroup_size:
               layout->num_driver_params = MAX2(layout->num_driver_params,
                                                IR3_DP_LOCAL_GROUP_SIZE_Z + 1);
               break;
            case nir_intrinsic_load_base_workgroup_id:
               layout->num_driver_params =
                  MAX2(layout->num_driver_params, IR3_DP_BASE_GROUP_Z + 1);
               break;
            case nir_intrinsic_load_subgroup_size:
               layout->num_driver_params =
                  MAX2(layout->num_driver_params, IR3_DP_SUBGROUP_SIZE + 1);
               break;
            case nir_intrinsic_load_subgroup_id_shift_ir3:
               layout->num_driver_params =
                  MAX2(layout->num_driver_params, IR3_DP_SUBGROUP_ID_SHIFT + 1);
               break;
            default:
               break;
            }
         }
      }
   }
}

/* Sets up the variant-dependent constant state for the ir3_shader.  Note
 * that it is also used from ir3_nir_analyze_ubo_ranges() to figure out the
 * maximum number of driver params that would eventually be used, to leave
 * space for this function to allocate the driver params.
 */
void
ir3_setup_const_state(nir_shader *nir, struct ir3_shader_variant *v,
                      struct ir3_const_state *const_state)
{
   struct ir3_compiler *compiler = v->shader->compiler;

   memset(&const_state->offsets, ~0, sizeof(const_state->offsets));

   ir3_nir_scan_driver_consts(compiler, nir, const_state);

   if ((compiler->gen < 5) && (v->shader->stream_output.num_outputs > 0)) {
      const_state->num_driver_params =
         MAX2(const_state->num_driver_params, IR3_DP_VTXCNT_MAX + 1);
   }

   const_state->num_ubos = nir->info.num_ubos;

   debug_assert((const_state->ubo_state.size % 16) == 0);
   unsigned constoff = const_state->ubo_state.size / 16;
   unsigned ptrsz = ir3_pointer_size(compiler);

   if (const_state->num_ubos > 0) {
      const_state->offsets.ubo = constoff;
      constoff += align(const_state->num_ubos * ptrsz, 4) / 4;
   }

   if (const_state->image_dims.count > 0) {
      unsigned cnt = const_state->image_dims.count;
      const_state->offsets.image_dims = constoff;
      constoff += align(cnt, 4) / 4;
   }

   if (v->type == MESA_SHADER_KERNEL) {
      const_state->offsets.kernel_params = constoff;
      constoff += align(v->shader->cs.req_input_mem, 4) / 4;
   }

   if (const_state->num_driver_params > 0) {
      /* num_driver_params in dwords.  we only need to align to vec4s for the
       * common case of immediate constant uploads, but for indirect dispatch
       * the constants may also be indirect and so we have to align the area in
       * const space to that requirement.
       */
      const_state->num_driver_params = align(const_state->num_driver_params, 4);
      unsigned upload_unit = 1;
      if (v->type == MESA_SHADER_COMPUTE ||
          (const_state->num_driver_params >= IR3_DP_VTXID_BASE)) {
         upload_unit = compiler->const_upload_unit;
      }

      /* offset cannot be 0 for vs params loaded by CP_DRAW_INDIRECT_MULTI */
      if (v->type == MESA_SHADER_VERTEX && compiler->gen >= 6)
         constoff = MAX2(constoff, 1);
      constoff = align(constoff, upload_unit);
      const_state->offsets.driver_param = constoff;

      constoff += align(const_state->num_driver_params / 4, upload_unit);
   }

   if ((v->type == MESA_SHADER_VERTEX) && (compiler->gen < 5) &&
       v->shader->stream_output.num_outputs > 0) {
      const_state->offsets.tfbo = constoff;
      constoff += align(IR3_MAX_SO_BUFFERS * ptrsz, 4) / 4;
   }

   switch (v->type) {
   case MESA_SHADER_VERTEX:
      const_state->offsets.primitive_param = constoff;
      constoff += 1;
      break;
   case MESA_SHADER_TESS_CTRL:
   case MESA_SHADER_TESS_EVAL:
      constoff = align(constoff - 1, 4) + 3;
      const_state->offsets.primitive_param = constoff;
      const_state->offsets.primitive_map = constoff + 5;
      constoff += 5 + DIV_ROUND_UP(v->input_size, 4);
      break;
   case MESA_SHADER_GEOMETRY:
      const_state->offsets.primitive_param = constoff;
      const_state->offsets.primitive_map = constoff + 1;
      constoff += 1 + DIV_ROUND_UP(v->input_size, 4);
      break;
   default:
      break;
   }

   const_state->offsets.immediate = constoff;

   assert(constoff <= ir3_max_const(v));
}
