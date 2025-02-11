/*
 * Copyright © 2018 Valve Corporation
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 */

#include "aco_ir.h"

#ifdef LLVM_AVAILABLE
#include "llvm/ac_llvm_util.h"

#include "llvm-c/Disassembler.h"
#include <llvm/ADT/StringRef.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#endif

#include <array>
#include <iomanip>
#include <vector>

namespace aco {
namespace {

/**
 * Determines the GPU type to use for CLRXdisasm
 */
const char*
to_clrx_device_name(chip_class cc, radeon_family family)
{
   switch (cc) {
   case GFX6:
      switch (family) {
      case CHIP_TAHITI: return "tahiti";
      case CHIP_PITCAIRN: return "pitcairn";
      case CHIP_VERDE: return "capeverde";
      case CHIP_OLAND: return "oland";
      case CHIP_HAINAN: return "hainan";
      default: return nullptr;
      }
   case GFX7:
      switch (family) {
      case CHIP_BONAIRE: return "bonaire";
      case CHIP_KAVERI: return "gfx700";
      case CHIP_HAWAII: return "hawaii";
      default: return nullptr;
      }
   case GFX8:
      switch (family) {
      case CHIP_TONGA: return "tonga";
      case CHIP_ICELAND: return "iceland";
      case CHIP_CARRIZO: return "carrizo";
      case CHIP_FIJI: return "fiji";
      case CHIP_STONEY: return "stoney";
      case CHIP_POLARIS10: return "polaris10";
      case CHIP_POLARIS11: return "polaris11";
      case CHIP_POLARIS12: return "polaris12";
      case CHIP_VEGAM: return "polaris11";
      default: return nullptr;
      }
   case GFX9:
      switch (family) {
      case CHIP_VEGA10: return "vega10";
      case CHIP_VEGA12: return "vega12";
      case CHIP_VEGA20: return "vega20";
      case CHIP_RAVEN: return "raven";
      default: return nullptr;
      }
   case GFX10:
      switch (family) {
      case CHIP_NAVI10: return "gfx1010";
      case CHIP_NAVI12: return "gfx1011";
      default: return nullptr;
      }
   case GFX10_3:
      return nullptr;
   default: unreachable("Invalid chip class!"); return nullptr;
   }
}

bool
print_asm_clrx(Program* program, std::vector<uint32_t>& binary, FILE* output)
{
#ifdef _WIN32
   return true;
#else
   char path[] = "/tmp/fileXXXXXX";
   char line[2048], command[128];
   FILE* p;
   int fd;

   const char* gpu_type = to_clrx_device_name(program->chip_class, program->family);

   /* Dump the binary into a temporary file. */
   fd = mkstemp(path);
   if (fd < 0)
      return true;

   for (uint32_t w : binary) {
      if (write(fd, &w, sizeof(w)) == -1)
         goto fail;
   }

   sprintf(command, "clrxdisasm --gpuType=%s -r %s", gpu_type, path);

   p = popen(command, "r");
   if (p) {
      if (!fgets(line, sizeof(line), p)) {
         fprintf(output, "clrxdisasm not found\n");
         pclose(p);
         goto fail;
      }

      do {
         fputs(line, output);
      } while (fgets(line, sizeof(line), p));

      pclose(p);
   }

   return false;

fail:
   close(fd);
   unlink(path);
   return true;
#endif
}

#ifdef LLVM_AVAILABLE
std::pair<bool, size_t>
disasm_instr(chip_class chip, LLVMDisasmContextRef disasm, uint32_t* binary, unsigned exec_size,
             size_t pos, char* outline, unsigned outline_size)
{
   /* mask out src2 on v_writelane_b32 */
   if (((chip == GFX8 || chip == GFX9) && (binary[pos] & 0xffff8000) == 0xd28a0000) ||
       (chip >= GFX10 && (binary[pos] & 0xffff8000) == 0xd7610000)) {
      binary[pos + 1] = binary[pos + 1] & 0xF803FFFF;
   }

   size_t l =
      LLVMDisasmInstruction(disasm, (uint8_t*)&binary[pos], (exec_size - pos) * sizeof(uint32_t),
                            pos * 4, outline, outline_size);

   if (chip >= GFX10 && l == 8 && ((binary[pos] & 0xffff0000) == 0xd7610000) &&
       ((binary[pos + 1] & 0x1ff) == 0xff)) {
      /* v_writelane with literal uses 3 dwords but llvm consumes only 2 */
      l += 4;
   }

   bool invalid = false;
   size_t size;
   if (!l &&
       ((chip >= GFX9 && (binary[pos] & 0xffff8000) == 0xd1348000) ||  /* v_add_u32_e64 + clamp */
        (chip >= GFX10 && (binary[pos] & 0xffff8000) == 0xd7038000) || /* v_add_u16_e64 + clamp */
        (chip <= GFX9 && (binary[pos] & 0xffff8000) == 0xd1268000) ||  /* v_add_u16_e64 + clamp */
        (chip >= GFX10 && (binary[pos] & 0xffff8000) == 0xd76d8000) || /* v_add3_u32 + clamp */
        (chip == GFX9 && (binary[pos] & 0xffff8000) == 0xd1ff8000)) /* v_add3_u32 + clamp */) {
      strcpy(outline, "\tinteger addition + clamp");
      bool has_literal = chip >= GFX10 && (((binary[pos + 1] & 0x1ff) == 0xff) ||
                                           (((binary[pos + 1] >> 9) & 0x1ff) == 0xff));
      size = 2 + has_literal;
   } else if (chip >= GFX10 && l == 4 && ((binary[pos] & 0xfe0001ff) == 0x020000f9)) {
      strcpy(outline, "\tv_cndmask_b32 + sdwa");
      size = 2;
   } else if (!l) {
      strcpy(outline, "(invalid instruction)");
      size = 1;
      invalid = true;
   } else {
      assert(l % 4 == 0);
      size = l / 4;
   }

   return std::make_pair(invalid, size);
}

bool
print_asm_llvm(Program* program, std::vector<uint32_t>& binary, unsigned exec_size, FILE* output)
{
   std::vector<bool> referenced_blocks(program->blocks.size());
   referenced_blocks[0] = true;
   for (Block& block : program->blocks) {
      for (unsigned succ : block.linear_succs)
         referenced_blocks[succ] = true;
   }

   std::vector<llvm::SymbolInfoTy> symbols;
   std::vector<std::array<char, 16>> block_names;
   block_names.reserve(program->blocks.size());
   for (Block& block : program->blocks) {
      if (!referenced_blocks[block.index])
         continue;
      std::array<char, 16> name;
      sprintf(name.data(), "BB%u", block.index);
      block_names.push_back(name);
      symbols.emplace_back(block.offset * 4,
                           llvm::StringRef(block_names[block_names.size() - 1].data()), 0);
   }

   const char* features = "";
   if (program->chip_class >= GFX10 && program->wave_size == 64) {
      features = "+wavefrontsize64";
   }

   LLVMDisasmContextRef disasm =
      LLVMCreateDisasmCPUFeatures("amdgcn-mesa-mesa3d", ac_get_llvm_processor_name(program->family),
                                  features, &symbols, 0, NULL, NULL);

   size_t pos = 0;
   bool invalid = false;
   unsigned next_block = 0;

   unsigned prev_size = 0;
   unsigned prev_pos = 0;
   unsigned repeat_count = 0;
   while (pos < exec_size) {
      bool new_block =
         next_block < program->blocks.size() && pos == program->blocks[next_block].offset;
      if (pos + prev_size <= exec_size && prev_pos != pos && !new_block &&
          memcmp(&binary[prev_pos], &binary[pos], prev_size * 4) == 0) {
         repeat_count++;
         pos += prev_size;
         continue;
      } else {
         if (repeat_count)
            fprintf(output, "\t(then repeated %u times)\n", repeat_count);
         repeat_count = 0;
      }

      while (next_block < program->blocks.size() && pos == program->blocks[next_block].offset) {
         if (referenced_blocks[next_block])
            fprintf(output, "BB%u:\n", next_block);
         next_block++;
      }

      char outline[1024];
      std::pair<bool, size_t> res = disasm_instr(program->chip_class, disasm, binary.data(),
                                                 exec_size, pos, outline, sizeof(outline));
      invalid |= res.first;

      fprintf(output, "%-60s ;", outline);

      for (unsigned i = 0; i < res.second; i++)
         fprintf(output, " %.8x", binary[pos + i]);
      fputc('\n', output);

      prev_size = res.second;
      prev_pos = pos;
      pos += res.second;
   }
   assert(next_block == program->blocks.size());

   LLVMDisasmDispose(disasm);

   if (program->constant_data.size()) {
      fputs("\n/* constant data */\n", output);
      for (unsigned i = 0; i < program->constant_data.size(); i += 32) {
         fprintf(output, "[%.6u]", i);
         unsigned line_size = std::min<size_t>(program->constant_data.size() - i, 32);
         for (unsigned j = 0; j < line_size; j += 4) {
            unsigned size = std::min<size_t>(program->constant_data.size() - (i + j), 4);
            uint32_t v = 0;
            memcpy(&v, &program->constant_data[i + j], size);
            fprintf(output, " %.8x", v);
         }
         fputc('\n', output);
      }
   }

   return invalid;
}
#endif /* LLVM_AVAILABLE */

} /* end namespace */

bool
check_print_asm_support(Program* program)
{
#ifdef LLVM_AVAILABLE
   if (program->chip_class >= GFX8) {
      /* LLVM disassembler only supports GFX8+ */
      return true;
   }
#endif

#ifndef _WIN32
   /* Check if CLRX disassembler binary is available and can disassemble the program */
   return to_clrx_device_name(program->chip_class, program->family) &&
          system("clrxdisasm --version") == 0;
#else
   return false;
#endif
}

/* Returns true on failure */
bool
print_asm(Program* program, std::vector<uint32_t>& binary, unsigned exec_size, FILE* output)
{
#ifdef LLVM_AVAILABLE
   if (program->chip_class >= GFX8) {
      return print_asm_llvm(program, binary, exec_size, output);
   }
#endif

   return print_asm_clrx(program, binary, output);
}

} // namespace aco
