
hsaco-0073.bin:	file format elf64-amdgpu
	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1151"

Disassembly of section .text:

0000000000001000 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8>:
	s_load_b256 s[8:15], s[0:1], 0x8                           // 000000001000: F40C0200 F8000008
	s_load_b256 s[16:23], s[0:1], 0x28                         // 000000001008: F40C0400 F8000028
	s_load_b128 s[4:7], s[0:1], 0x48                           // 000000001010: F4080100 F8000048
	v_and_b32_e32 v1, 63, v0                                   // 000000001018: 360200BF
	v_lshrrev_b32_e32 v2, 6, v0                                // 00000000101C: 32040086
	v_mov_b32_e32 v4, 0                                        // 000000001020: 7E080280
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)// 000000001024: BF870151
	v_mov_b32_e32 v5, v4                                       // 000000001028: 7E0A0304
	v_mov_b32_e32 v3, 0xf149f2ca                               // 00000000102C: 7E0602FF F149F2CA
	v_and_b32_e32 v6, 31, v0                                   // 000000001034: 360C009F
	v_lshl_add_u32 v7, s2, 6, v1                               // 000000001038: D6460007 04050C02
	v_mov_b32_e32 v8, 0x4510                                   // 000000001040: 7E1002FF 00004510
	v_lshlrev_b32_e32 v7, 1, v7                                // 000000001048: 300E0E81
	s_waitcnt lgkmcnt(0)                                       // 00000000104C: BF89FC07
	global_load_d16_b16 v7, v7, s[14:15]                       // 000000001050: DC820000 070E0007
	s_waitcnt vmcnt(0)                                         // 000000001058: BF8903F7
	v_cvt_f32_f16_e64 v7, v7.l                                 // 00000000105C: D58B0007 02010107
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001064: BF870091
	v_max_f32_dpp v9, v7, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001068: 20120EFA FF08B107
	v_max_f32_dpp v9, v9, v9 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001070: 201212FA FF084E09
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001078: BF870091
	v_max_f32_dpp v9, v9, v9 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 00000000107C: 201212FA FF094109
	v_max_f32_dpp v9, v9, v9 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001084: 201212FA FF094009
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)// 00000000108C: BF8701B1
	v_permlanex16_b32 v10, v9, 0, 0                            // 000000001090: D65C000A 02010109
	v_lshrrev_b32_e32 v11, 5, v0                               // 000000001098: 32160085
	v_cmp_lt_u32_e64 s0, v6, 1                                 // 00000000109C: D4490000 02010306
	v_max_f32_e32 v6, v9, v10                                  // 0000000010A4: 200C1509
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000010A8: BF870193
	v_lshlrev_b32_e32 v9, 2, v11                               // 0000000010AC: 30121682
	s_and_saveexec_b64 s[0:1], s[0:1]                          // 0000000010B0: BE802100
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000010B4: BF870001
	v_add_nc_u32_e32 v9, v8, v9                                // 0000000010B8: 4A121308
	ds_store_b32 v9, v6                                        // 0000000010BC: D8340000 00000609
	s_mov_b64 exec, s[0:1]                                     // 0000000010C4: BEFE0100
	s_waitcnt lgkmcnt(0)                                       // 0000000010C8: BF89FC07
	s_barrier                                                  // 0000000010CC: BFBD0000
	v_cmp_lt_u32_e64 s0, v0, 1                                 // 0000000010D0: D4490000 02010300
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000010D8: BF870001
	s_and_saveexec_b64 s[0:1], s[0:1]                          // 0000000010DC: BE802100
	ds_load_b32 v6, v8                                         // 0000000010E0: D8D80000 06000008
	v_add_nc_u32_e32 v9, 4, v8                                 // 0000000010E8: 4A121084
	v_add_nc_u32_e32 v10, 8, v8                                // 0000000010EC: 4A141088
	v_add_nc_u32_e32 v11, 12, v8                               // 0000000010F0: 4A16108C
	v_add_nc_u32_e32 v12, 16, v8                               // 0000000010F4: 4A181090
	v_add_nc_u32_e32 v13, 20, v8                               // 0000000010F8: 4A1A1094
	ds_load_b32 v9, v9                                         // 0000000010FC: D8D80000 09000009
	ds_load_b32 v10, v10                                       // 000000001104: D8D80000 0A00000A
	ds_load_b32 v11, v11                                       // 00000000110C: D8D80000 0B00000B
	ds_load_b32 v12, v12                                       // 000000001114: D8D80000 0C00000C
	ds_load_b32 v13, v13                                       // 00000000111C: D8D80000 0D00000D
	v_add_nc_u32_e32 v14, 24, v8                               // 000000001124: 4A1C1098
	v_add_nc_u32_e32 v15, 28, v8                               // 000000001128: 4A1E109C
	s_waitcnt lgkmcnt(4)                                       // 00000000112C: BF89FC47
	v_max_f32_e32 v6, v6, v9                                   // 000000001130: 200C1306
	ds_load_b32 v9, v14                                        // 000000001134: D8D80000 0900000E
	ds_load_b32 v14, v15                                       // 00000000113C: D8D80000 0E00000F
	s_waitcnt lgkmcnt(5)                                       // 000000001144: BF89FC57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001148: BF8700A1
	v_max_f32_e32 v6, v6, v10                                  // 00000000114C: 200C1506
	s_waitcnt lgkmcnt(4)                                       // 000000001150: BF89FC47
	v_max_f32_e32 v6, v6, v11                                  // 000000001154: 200C1706
	s_waitcnt lgkmcnt(3)                                       // 000000001158: BF89FC37
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000115C: BF8700A1
	v_max_f32_e32 v6, v6, v12                                  // 000000001160: 200C1906
	s_waitcnt lgkmcnt(2)                                       // 000000001164: BF89FC27
	v_max_f32_e32 v6, v6, v13                                  // 000000001168: 200C1B06
	s_waitcnt lgkmcnt(1)                                       // 00000000116C: BF89FC17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001170: BF8700A1
	v_max_f32_e32 v6, v6, v9                                   // 000000001174: 200C1306
	s_waitcnt lgkmcnt(0)                                       // 000000001178: BF89FC07
	v_max_f32_e32 v6, v6, v14                                  // 00000000117C: 200C1D06
	s_mov_b64 exec, s[0:1]                                     // 000000001180: BEFE0100
	v_cmp_lt_u32_e64 s0, v0, 1                                 // 000000001184: D4490000 02010300
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000118C: BF870001
	s_and_saveexec_b64 s[0:1], s[0:1]                          // 000000001190: BE802100
	ds_store_b32 v8, v6                                        // 000000001194: D8340000 00000608
	s_mov_b64 exec, s[0:1]                                     // 00000000119C: BEFE0100
	s_waitcnt lgkmcnt(0)                                       // 0000000011A0: BF89FC07
	s_barrier                                                  // 0000000011A4: BFBD0000
	ds_load_b32 v6, v8                                         // 0000000011A8: D8D80000 06000008
	s_mov_b64 s[0:1], exec                                     // 0000000011B0: BE80017E
	s_waitcnt lgkmcnt(0)                                       // 0000000011B4: BF89FC07
	v_cmp_gt_f32_e64 s24, v6, v3                               // 0000000011B8: D4140018 02020706
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011C0: BF870001
	s_and_b64 s[0:1], s[24:25], s[0:1]                         // 0000000011C4: 8B800018
	s_cbranch_scc0 1413                                        // 0000000011C8: BFA10585 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x17e0>
	v_lshrrev_b32_e32 v1, lit(0x7), v0                         // 0000000011CC: 320200FF 00000007
	v_and_b32_e32 v2, 0x7f, v0                                 // 0000000011D4: 360400FF 0000007F
	v_add_nc_u32_e32 v3, 0x100, v0                             // 0000000011DC: 4A0600FF 00000100
	s_mov_b32 s0, 0x100                                        // 0000000011E4: BE8000FF 00000100
	v_lshl_add_u32 v4, s0, 1, v0                               // 0000000011EC: D6460004 04010200
	s_lshl_b32 s1, s3, 12                                      // 0000000011F4: 84018C03
	s_delay_alu instid0(VALU_DEP_3) | instid1(VALU_DEP_4)      // 0000000011F8: BF870203
	v_lshl_add_u32 v5, v1, 7, v2                               // 0000000011FC: D6460005 04090F01
	v_mad_u32_u24 v1, v1, 0x88, v2                             // 000000001204: D60B0001 0409FF01 00000088
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 000000001210: BF870234
	v_lshrrev_b32_e32 v2, lit(0x7), v3                         // 000000001214: 320406FF 00000007
	v_and_b32_e32 v3, 0x7f, v3                                 // 00000000121C: 360606FF 0000007F
	s_mov_b32 s24, 0                                           // 000000001224: BE980080
	v_lshlrev_b32_e32 v5, 2, v5                                // 000000001228: 300A0A82
	s_add_u32 s26, s8, s1                                      // 00000000122C: 801A0108
	s_addc_u32 s27, s9, s24                                    // 000000001230: 821B1809
	s_delay_alu instid0(VALU_DEP_2) | instid1(VALU_DEP_3)      // 000000001234: BF870182
	v_lshl_add_u32 v6, v2, 7, v3                               // 000000001238: D6460006 040D0F02
	v_mad_u32_u24 v2, v2, 0x88, v3                             // 000000001240: D60B0002 040DFF02 00000088
	global_load_b32 v3, v5, s[26:27]                           // 00000000124C: DC520000 031A0005
	v_lshrrev_b32_e32 v5, lit(0x7), v4                         // 000000001254: 320A08FF 00000007
	v_and_b32_e32 v4, 0x7f, v4                                 // 00000000125C: 360808FF 0000007F
	v_lshlrev_b32_e32 v6, 2, v6                                // 000000001264: 300C0C82
	s_mov_b32 s1, 3                                            // 000000001268: BE810083
	v_lshl_add_u32 v8, s1, 8, v0                               // 00000000126C: D6460008 04011001
	s_lshl_b32 s1, s3, 12                                      // 000000001274: 84018C03
	s_waitcnt vmcnt(0)                                         // 000000001278: BF8903F7
	s_add_u32 s26, s8, s1                                      // 00000000127C: 801A0108
	s_delay_alu instid0(VALU_DEP_3) | instid1(VALU_DEP_4)      // 000000001280: BF870203
	v_lshl_add_u32 v9, v5, 7, v4                               // 000000001284: D6460009 04110F05
	v_mad_u32_u24 v4, v5, 0x88, v4                             // 00000000128C: D60B0004 0411FF05 00000088
	s_addc_u32 s27, s9, s24                                    // 000000001298: 821B1809
	global_load_b32 v5, v6, s[26:27]                           // 00000000129C: DC520000 051A0006
	v_lshrrev_b32_e32 v6, lit(0x7), v8                         // 0000000012A4: 320C10FF 00000007
	v_and_b32_e32 v8, 0x7f, v8                                 // 0000000012AC: 361010FF 0000007F
	v_lshlrev_b32_e32 v9, 2, v9                                // 0000000012B4: 30121282
	s_lshl_b32 s1, s3, 12                                      // 0000000012B8: 84018C03
	s_waitcnt vmcnt(0)                                         // 0000000012BC: BF8903F7
	s_add_u32 s26, s8, s1                                      // 0000000012C0: 801A0108
	s_addc_u32 s27, s9, s24                                    // 0000000012C4: 821B1809
	s_delay_alu instid0(VALU_DEP_2) | instid1(VALU_DEP_3)      // 0000000012C8: BF870182
	v_lshl_add_u32 v10, v6, 7, v8                              // 0000000012CC: D646000A 04210F06
	global_load_b32 v9, v9, s[26:27]                           // 0000000012D4: DC520000 091A0009
	v_mad_u32_u24 v6, v6, 0x88, v8                             // 0000000012DC: D60B0006 0421FF06 00000088
	v_mul_f32_e32 v3, 0x3db504f3, v3                           // 0000000012E8: 100606FF 3DB504F3
	s_lshl_b32 s1, s3, 12                                      // 0000000012F0: 84018C03
	v_lshlrev_b32_e32 v8, 2, v10                               // 0000000012F4: 30101482
	s_waitcnt vmcnt(0)                                         // 0000000012F8: BF8903F7
	s_add_u32 s26, s8, s1                                      // 0000000012FC: 801A0108
	s_addc_u32 s27, s9, s24                                    // 000000001300: 821B1809
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001304: BF870002
	v_cvt_f16_f32_e32 v3.l, v3                                 // 000000001308: 7E061503
	v_mul_f32_e32 v5, 0x3db504f3, v5                           // 00000000130C: 100A0AFF 3DB504F3
	global_load_b32 v8, v8, s[26:27]                           // 000000001314: DC520000 081A0008
	v_lshlrev_b32_e32 v1, 1, v1                                // 00000000131C: 30020281
	v_lshlrev_b32_e32 v2, 1, v2                                // 000000001320: 30040481
	v_bfe_u32 v3, v3, 0, 16                                    // 000000001324: D6100003 02410103
	v_cvt_f16_f32_e32 v5.l, v5                                 // 00000000132C: 7E0A1505
	v_lshrrev_b32_e32 v10, 6, v0                               // 000000001330: 32140086
	v_lshlrev_b32_e32 v4, 1, v4                                // 000000001334: 30080881
	v_mul_f32_e32 v9, 0x3db504f3, v9                           // 000000001338: 101212FF 3DB504F3
	ds_store_b16 v1, v3                                        // 000000001340: D87C0000 00000301
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001348: BF870214
	v_bfe_u32 v1, v5, 0, 16                                    // 00000000134C: D6100001 02410105
	v_lshlrev_b32_e32 v3, lit(0x2), v10                        // 000000001354: 300614FF 00000002
	v_lshl_add_u32 v5, s0, 2, v0                               // 00000000135C: D6460005 04010400
	s_delay_alu instid0(VALU_DEP_4)                            // 000000001364: BF870004
	v_cvt_f16_f32_e32 v9.l, v9                                 // 000000001368: 7E121509
	v_lshlrev_b32_e32 v6, 1, v6                                // 00000000136C: 300C0C81
	v_and_b32_e32 v11, 63, v0                                  // 000000001370: 361600BF
	v_add_nc_u32_e32 v12, lit(0x1), v3                         // 000000001374: 4A1806FF 00000001
	s_waitcnt vmcnt(0)                                         // 00000000137C: BF8903F7
	v_mul_f32_e32 v8, 0x3db504f3, v8                           // 000000001380: 101010FF 3DB504F3
	v_bfe_u32 v9, v9, 0, 16                                    // 000000001388: D6100009 02410109
	v_add_nc_u32_e32 v13, lit(0x2), v3                         // 000000001390: 4A1A06FF 00000002
	v_cmp_lt_u32_e64 s0, v11, 32                               // 000000001398: D4490000 0201410B
	v_add_nc_u32_e32 v14, lit(0x3), v3                         // 0000000013A0: 4A1C06FF 00000003
	v_cvt_f16_f32_e32 v8.l, v8                                 // 0000000013A8: 7E101508
	v_lshrrev_b32_e32 v15, lit(0x7), v5                        // 0000000013AC: 321E0AFF 00000007
	v_and_b32_e32 v5, 0x7f, v5                                 // 0000000013B4: 360A0AFF 0000007F
	v_lshl_add_u32 v3, s3, 3, v3                               // 0000000013BC: D6460003 040D0603
	v_lshl_add_u32 v12, s3, 3, v12                             // 0000000013C4: D646000C 04310603
	v_bfe_u32 v8, v8, 0, 16                                    // 0000000013CC: D6100008 02410108
	v_lshl_add_u32 v13, s3, 3, v13                             // 0000000013D4: D646000D 04350603
	v_lshl_add_u32 v14, s3, 3, v14                             // 0000000013DC: D646000E 04390603
	v_lshl_add_u32 v16, s3, 3, v15                             // 0000000013E4: D6460010 043D0603
	v_cmp_lt_u32_e64 s24, v3, 32                               // 0000000013EC: D4490018 02014103
	v_cmp_lt_u32_e64 s28, v12, 32                              // 0000000013F4: D449001C 0201410C
	v_cmp_lt_u32_e64 s30, v13, 32                              // 0000000013FC: D449001E 0201410D
	v_cmp_lt_u32_e64 s32, v14, 32                              // 000000001404: D4490020 0201410E
	v_cmp_lt_u32_e64 s34, v16, 32                              // 00000000140C: D4490022 02014110
	s_mov_b32 s36, 7                                           // 000000001414: BEA40087
	ds_store_b16 v2, v1                                        // 000000001418: D87C0000 00000102
	ds_store_b16 v4, v9                                        // 000000001420: D87C0000 00000904
	ds_store_b16 v6, v8                                        // 000000001428: D87C0000 00000806
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001430: BF870001
	s_and_saveexec_b64 s[38:39], s[34:35]                      // 000000001434: BEA62122
	s_cbranch_scc0 1981                                        // 000000001438: BFA107BD <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2330>
	v_lshl_add_u32 v1, v15, 7, v5                              // 00000000143C: D6460001 04150F0F
	s_lshl_b32 s37, s3, 12                                     // 000000001444: 84258C03
	s_add_u32 s26, s8, s37                                     // 000000001448: 801A2508
	s_mov_b32 s27, 0                                           // 00000000144C: BE9B0080
	s_addc_u32 s27, s9, s27                                    // 000000001450: 821B1B09
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001454: BF8700C1
	v_lshlrev_b32_e32 v1, 2, v1                                // 000000001458: 30020282
	global_load_b32 v1, v1, s[26:27]                           // 00000000145C: DC520000 011A0001
	s_waitcnt vmcnt(0)                                         // 000000001464: BF8903F7
	v_mul_f32_e32 v1, 0x3db504f3, v1                           // 000000001468: 100202FF 3DB504F3
	v_cvt_f16_f32_e32 v1.l, v1                                 // 000000001470: 7E021501
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001474: BF870001
	v_bfe_u32 v1, v1, 0, 16                                    // 000000001478: D6100001 02410101
	s_branch 1958                                              // 000000001480: BFA007A6 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x231c>
	s_mov_b64 exec, s[38:39]                                   // 000000001484: BEFE0126
	v_cndmask_b32_e64 v1, 0, v1, s34                           // 000000001488: D5010001 008A0280
	s_mov_b32 s34, 5                                           // 000000001490: BEA20085
	v_lshl_add_u32 v2, s34, 8, v0                              // 000000001494: D6460002 04011022
	v_mad_u32_u24 v3, v15, 0x88, v5                            // 00000000149C: D60B0003 0415FF0F 00000088
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 0000000014A8: BF8701A2
	v_lshrrev_b32_e32 v4, lit(0x7), v2                         // 0000000014AC: 320804FF 00000007
	v_and_b32_e32 v2, 0x7f, v2                                 // 0000000014B4: 360404FF 0000007F
	v_lshlrev_b32_e32 v3, 1, v3                                // 0000000014BC: 30060681
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000014C0: BF8700A3
	v_lshl_add_u32 v5, s3, 3, v4                               // 0000000014C4: D6460005 04110603
	ds_store_b16 v3, v1                                        // 0000000014CC: D87C0000 00000103
	v_cmp_lt_u32_e64 s34, v5, 32                               // 0000000014D4: D4490022 02014105
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000014DC: BF870001
	s_and_saveexec_b64 s[38:39], s[34:35]                      // 0000000014E0: BEA62122
	s_cbranch_scc0 1948                                        // 0000000014E4: BFA1079C <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2358>
	v_lshl_add_u32 v1, v4, 7, v2                               // 0000000014E8: D6460001 04090F04
	s_lshl_b32 s37, s3, 12                                     // 0000000014F0: 84258C03
	s_add_u32 s26, s8, s37                                     // 0000000014F4: 801A2508
	s_mov_b32 s27, 0                                           // 0000000014F8: BE9B0080
	s_addc_u32 s27, s9, s27                                    // 0000000014FC: 821B1B09
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001500: BF8700C1
	v_lshlrev_b32_e32 v1, 2, v1                                // 000000001504: 30020282
	global_load_b32 v1, v1, s[26:27]                           // 000000001508: DC520000 011A0001
	s_waitcnt vmcnt(0)                                         // 000000001510: BF8903F7
	v_mul_f32_e32 v1, 0x3db504f3, v1                           // 000000001514: 100202FF 3DB504F3
	v_cvt_f16_f32_e32 v1.l, v1                                 // 00000000151C: 7E021501
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001520: BF870001
	v_bfe_u32 v1, v1, 0, 16                                    // 000000001524: D6100001 02410101
	s_branch 1925                                              // 00000000152C: BFA00785 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2344>
	s_mov_b64 exec, s[38:39]                                   // 000000001530: BEFE0126
	v_cndmask_b32_e64 v1, 0, v1, s34                           // 000000001534: D5010001 008A0280
	s_mov_b32 s34, 6                                           // 00000000153C: BEA20086
	v_lshl_add_u32 v3, s34, 8, v0                              // 000000001540: D6460003 04011022
	v_mad_u32_u24 v2, v4, 0x88, v2                             // 000000001548: D60B0002 0409FF04 00000088
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000001554: BF8701A2
	v_lshrrev_b32_e32 v4, lit(0x7), v3                         // 000000001558: 320806FF 00000007
	v_and_b32_e32 v3, 0x7f, v3                                 // 000000001560: 360606FF 0000007F
	v_lshlrev_b32_e32 v2, 1, v2                                // 000000001568: 30040481
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000156C: BF8700A3
	v_lshl_add_u32 v5, s3, 3, v4                               // 000000001570: D6460005 04110603
	ds_store_b16 v2, v1                                        // 000000001578: D87C0000 00000102
	v_cmp_lt_u32_e64 s34, v5, 32                               // 000000001580: D4490022 02014105
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001588: BF870001
	s_and_saveexec_b64 s[38:39], s[34:35]                      // 00000000158C: BEA62122
	s_cbranch_scc0 1915                                        // 000000001590: BFA1077B <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2380>
	v_lshl_add_u32 v1, v4, 7, v3                               // 000000001594: D6460001 040D0F04
	s_lshl_b32 s37, s3, 12                                     // 00000000159C: 84258C03
	s_add_u32 s26, s8, s37                                     // 0000000015A0: 801A2508
	s_mov_b32 s27, 0                                           // 0000000015A4: BE9B0080
	s_addc_u32 s27, s9, s27                                    // 0000000015A8: 821B1B09
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000015AC: BF8700C1
	v_lshlrev_b32_e32 v1, 2, v1                                // 0000000015B0: 30020282
	global_load_b32 v1, v1, s[26:27]                           // 0000000015B4: DC520000 011A0001
	s_waitcnt vmcnt(0)                                         // 0000000015BC: BF8903F7
	v_mul_f32_e32 v1, 0x3db504f3, v1                           // 0000000015C0: 100202FF 3DB504F3
	v_cvt_f16_f32_e32 v1.l, v1                                 // 0000000015C8: 7E021501
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000015CC: BF870001
	v_bfe_u32 v1, v1, 0, 16                                    // 0000000015D0: D6100001 02410101
	s_branch 1892                                              // 0000000015D8: BFA00764 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x236c>
	s_mov_b64 exec, s[38:39]                                   // 0000000015DC: BEFE0126
	v_cndmask_b32_e64 v1, 0, v1, s34                           // 0000000015E0: D5010001 008A0280
	v_lshl_add_u32 v2, s36, 8, v0                              // 0000000015E8: D6460002 04011024
	v_mad_u32_u24 v3, v4, 0x88, v3                             // 0000000015F0: D60B0003 040DFF04 00000088
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 0000000015FC: BF8701A2
	v_lshrrev_b32_e32 v4, lit(0x7), v2                         // 000000001600: 320804FF 00000007
	v_and_b32_e32 v2, 0x7f, v2                                 // 000000001608: 360404FF 0000007F
	v_lshlrev_b32_e32 v3, 1, v3                                // 000000001610: 30060681
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001614: BF8700A3
	v_lshl_add_u32 v5, s3, 3, v4                               // 000000001618: D6460005 04110603
	ds_store_b16 v3, v1                                        // 000000001620: D87C0000 00000103
	v_cmp_lt_u32_e64 s34, v5, 32                               // 000000001628: D4490022 02014105
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001630: BF870001
	s_and_saveexec_b64 s[36:37], s[34:35]                      // 000000001634: BEA42122
	s_cbranch_scc0 1883                                        // 000000001638: BFA1075B <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x23a8>
	v_lshl_add_u32 v1, v4, 7, v2                               // 00000000163C: D6460001 04090F04
	s_lshl_b32 s38, s3, 12                                     // 000000001644: 84268C03
	s_add_u32 s26, s8, s38                                     // 000000001648: 801A2608
	s_mov_b32 s27, 0                                           // 00000000164C: BE9B0080
	s_addc_u32 s27, s9, s27                                    // 000000001650: 821B1B09
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001654: BF8700C1
	v_lshlrev_b32_e32 v1, 2, v1                                // 000000001658: 30020282
	global_load_b32 v1, v1, s[26:27]                           // 00000000165C: DC520000 011A0001
	s_waitcnt vmcnt(0)                                         // 000000001664: BF8903F7
	v_mul_f32_e32 v1, 0x3db504f3, v1                           // 000000001668: 100202FF 3DB504F3
	v_cvt_f16_f32_e32 v1.l, v1                                 // 000000001670: 7E021501
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001674: BF870001
	v_bfe_u32 v1, v1, 0, 16                                    // 000000001678: D6100001 02410101
	s_branch 1860                                              // 000000001680: BFA00744 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2394>
	s_mov_b64 exec, s[36:37]                                   // 000000001684: BEFE0124
	v_cndmask_b32_e64 v1, 0, v1, s34                           // 000000001688: D5010001 008A0280
	v_mad_u32_u24 v2, v4, 0x88, v2                             // 000000001690: D60B0002 0409FF04 00000088
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000169C: BF870001
	v_lshlrev_b32_e32 v2, 1, v2                                // 0000000016A0: 30040481
	ds_store_b16 v2, v1                                        // 0000000016A4: D87C0000 00000102
	s_waitcnt lgkmcnt(0)                                       // 0000000016AC: BF89FC07
	s_barrier                                                  // 0000000016B0: BFBD0000
	v_and_b32_e32 v1, 63, v0                                   // 0000000016B4: 360200BF
	v_lshlrev_b32_e32 v2, 14, v10                              // 0000000016B8: 3004148E
	s_lshl_b32 s8, s2, 16                                      // 0000000016BC: 84089002
	s_lshl_b32 s9, s3, 8                                       // 0000000016C0: 84098803
	s_lshl_b32 s34, s2, 16                                     // 0000000016C4: 84229002
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000016C8: BF870112
	v_and_b32_e32 v3, 15, v1                                   // 0000000016CC: 3606028F
	v_add_nc_u32_e32 v4, s8, v2                                // 0000000016D0: 4A080408
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000016D4: BF870001
	v_add_nc_u32_e32 v4, s9, v4                                // 0000000016D8: 4A080809
	v_add_nc_u32_e32 v5, s34, v2                               // 0000000016DC: 4A0A0422
	v_lshrrev_b32_e32 v1, 4, v1                                // 0000000016E0: 32020284
	v_mul_u32_u24_e32 v6, 0x110, v3                            // 0000000016E4: 160C06FF 00000110
	v_lshlrev_b32_e32 v8, 10, v3                               // 0000000016EC: 3010068A
	s_lshl_b32 s8, s3, 8                                       // 0000000016F0: 84088803
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000016F4: BF870004
	v_add_nc_u32_e32 v5, s8, v5                                // 0000000016F8: 4A0A0A08
	s_lshl_b32 s8, s2, 16                                      // 0000000016FC: 84089002
	ds_load_b128 v[16:19], v6                                  // 000000001700: DBFC0000 10000006
	ds_load_b128 v[20:23], v6 offset:16                        // 000000001708: DBFC0010 14000006
	ds_load_b128 v[24:27], v6 offset:32                        // 000000001710: DBFC0020 18000006
	ds_load_b128 v[28:31], v6 offset:48                        // 000000001718: DBFC0030 1C000006
	ds_load_b128 v[32:35], v6 offset:64                        // 000000001720: DBFC0040 20000006
	ds_load_b128 v[36:39], v6 offset:80                        // 000000001728: DBFC0050 24000006
	ds_load_b128 v[40:43], v6 offset:96                        // 000000001730: DBFC0060 28000006
	ds_load_b128 v[44:47], v6 offset:112                       // 000000001738: DBFC0070 2C000006
	ds_load_b128 v[12:15], v6 offset:128                       // 000000001740: DBFC0080 0C000006
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001748: BF870112
	v_add_nc_u32_e32 v4, v4, v8                                // 00000000174C: 4A081104
	v_add_nc_u32_e32 v5, v5, v8                                // 000000001750: 4A0A1105
	ds_load_b128 v[48:51], v6 offset:144                       // 000000001754: DBFC0090 30000006
	s_waitcnt lgkmcnt(0)                                       // 00000000175C: BF89FC07
	v_mov_b32_e32 v52, v48                                     // 000000001760: 7E680330
	v_mov_b32_e32 v53, v49                                     // 000000001764: 7E6A0331
	v_mov_b32_e32 v54, v50                                     // 000000001768: 7E6C0332
	v_mov_b32_e32 v55, v51                                     // 00000000176C: 7E6E0333
	v_mov_b32_e32 v48, v12                                     // 000000001770: 7E60030C
	v_mov_b32_e32 v49, v13                                     // 000000001774: 7E62030D
	v_mov_b32_e32 v50, v14                                     // 000000001778: 7E64030E
	v_mov_b32_e32 v51, v15                                     // 00000000177C: 7E66030F
	v_add_nc_u32_e32 v9, s8, v2                                // 000000001780: 4A120408
	s_lshl_b32 s8, s3, 8                                       // 000000001784: 84088803
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001788: BF870091
	v_add_nc_u32_e32 v9, s8, v9                                // 00000000178C: 4A121208
	v_add_nc_u32_e32 v9, v9, v8                                // 000000001790: 4A121109
	s_lshl_b32 s8, s2, 16                                      // 000000001794: 84089002
	v_add_nc_u32_e32 v12, s8, v2                               // 000000001798: 4A180408
	s_lshl_b32 s8, s3, 8                                       // 00000000179C: 84088803
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017A0: BF870091
	v_add_nc_u32_e32 v12, s8, v12                              // 0000000017A4: 4A181808
	v_add_nc_u32_e32 v12, v12, v8                              // 0000000017A8: 4A18110C
	s_lshl_b32 s8, s2, 16                                      // 0000000017AC: 84089002
	v_add_nc_u32_e32 v13, s8, v2                               // 0000000017B0: 4A1A0408
	s_lshl_b32 s8, s3, 8                                       // 0000000017B4: 84088803
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017B8: BF870091
	v_add_nc_u32_e32 v13, s8, v13                              // 0000000017BC: 4A1A1A08
	v_add_nc_u32_e32 v13, v13, v8                              // 0000000017C0: 4A1A110D
	s_lshl_b32 s8, s2, 16                                      // 0000000017C4: 84089002
	v_add_nc_u32_e32 v14, s8, v2                               // 0000000017C8: 4A1C0408
	s_lshl_b32 s8, s3, 8                                       // 0000000017CC: 84088803
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017D0: BF870091
	v_add_nc_u32_e32 v14, s8, v14                              // 0000000017D4: 4A1C1C08
	v_add_nc_u32_e32 v14, v14, v8                              // 0000000017D8: 4A1C110E
	s_lshl_b32 s8, s2, 16                                      // 0000000017DC: 84089002
	v_add_nc_u32_e32 v15, s8, v2                               // 0000000017E0: 4A1E0408
	s_lshl_b32 s8, s3, 8                                       // 0000000017E4: 84088803
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017E8: BF870091
	v_add_nc_u32_e32 v15, s8, v15                              // 0000000017EC: 4A1E1E08
	v_add_nc_u32_e32 v15, v15, v8                              // 0000000017F0: 4A1E110F
	global_load_b128 v[56:59], v4, s[10:11]                    // 0000000017F4: DC5E0000 380A0004
	global_load_b128 v[60:63], v4, s[10:11] offset:16          // 0000000017FC: DC5E0010 3C0A0004
	s_waitcnt vmcnt(0)                                         // 000000001804: BF8903F7
	v_wmma_f32_16x16x16_f16 v[64:71], v[56:63], v[16:23], 0    // 000000001808: CC404040 1A022138
	global_load_b128 v[16:19], v5, s[10:11] offset:32          // 000000001810: DC5E0020 100A0005
	global_load_b128 v[20:23], v5, s[10:11] offset:48          // 000000001818: DC5E0030 140A0005
	s_waitcnt vmcnt(0)                                         // 000000001820: BF8903F7
	v_wmma_f32_16x16x16_f16 v[64:71], v[16:23], v[24:31], v[64:71]// 000000001824: CC404040 1D023110
	global_load_b128 v[16:19], v9, s[10:11] offset:64          // 00000000182C: DC5E0040 100A0009
	global_load_b128 v[20:23], v9, s[10:11] offset:80          // 000000001834: DC5E0050 140A0009
	global_load_b128 v[24:27], v12, s[10:11] offset:96         // 00000000183C: DC5E0060 180A000C
	s_waitcnt vmcnt(1)                                         // 000000001844: BF8907F7
	v_wmma_f32_16x16x16_f16 v[64:71], v[16:23], v[32:39], v[64:71]// 000000001848: CC404040 1D024110
	global_load_b128 v[28:31], v12, s[10:11] offset:112        // 000000001850: DC5E0070 1C0A000C
	global_load_b128 v[16:19], v13, s[10:11] offset:128        // 000000001858: DC5E0080 100A000D
	global_load_b128 v[20:23], v13, s[10:11] offset:144        // 000000001860: DC5E0090 140A000D
	global_load_b128 v[32:35], v14, s[10:11] offset:160        // 000000001868: DC5E00A0 200A000E
	s_waitcnt vmcnt(3)                                         // 000000001870: BF890FF7
	v_wmma_f32_16x16x16_f16 v[64:71], v[24:31], v[40:47], v[64:71]// 000000001874: CC404040 1D025118
	global_load_b128 v[36:39], v14, s[10:11] offset:176        // 00000000187C: DC5E00B0 240A000E
	global_load_b128 v[24:27], v15, s[10:11] offset:192        // 000000001884: DC5E00C0 180A000F
	global_load_b128 v[28:31], v15, s[10:11] offset:208        // 00000000188C: DC5E00D0 1C0A000F
	ds_load_b128 v[40:43], v6 offset:160                       // 000000001894: DBFC00A0 28000006
	ds_load_b128 v[44:47], v6 offset:176                       // 00000000189C: DBFC00B0 2C000006
	s_waitcnt vmcnt(4)                                         // 0000000018A4: BF8913F7
	v_wmma_f32_16x16x16_f16 v[64:71], v[16:23], v[48:55], v[64:71]// 0000000018A8: CC404040 1D026110
	ds_load_b128 v[16:19], v6 offset:192                       // 0000000018B0: DBFC00C0 10000006
	ds_load_b128 v[20:23], v6 offset:208                       // 0000000018B8: DBFC00D0 14000006
	ds_load_b128 v[48:51], v6 offset:224                       // 0000000018C0: DBFC00E0 30000006
	ds_load_b128 v[52:55], v6 offset:240                       // 0000000018C8: DBFC00F0 34000006
	s_lshl_b32 s8, s2, 16                                      // 0000000018D0: 84089002
	v_add_nc_u32_e32 v2, s8, v2                                // 0000000018D4: 4A040408
	s_lshl_b32 s8, s3, 8                                       // 0000000018D8: 84088803
	v_mul_lo_u32 v4, v10, 0x600                                // 0000000018DC: D72C0004 0201FF0A 00000600
	v_lshlrev_b32_e32 v5, 2, v3                                // 0000000018E8: 300A0682
	v_mul_u32_u24_e32 v6, 0x60, v1                             // 0000000018EC: 160C02FF 00000060
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 0000000018F4: BF870134
	v_add_nc_u32_e32 v2, s8, v2                                // 0000000018F8: 4A040408
	s_waitcnt vmcnt(2) lgkmcnt(4)                              // 0000000018FC: BF890847
	v_wmma_f32_16x16x16_f16 v[64:71], v[32:39], v[40:47], v[64:71]// 000000001900: CC404040 1D025120
	v_add_nc_u32_e32 v2, v2, v8                                // 000000001908: 4A041102
	v_add_nc_u32_e32 v4, v4, v5                                // 00000000190C: 4A080B04
	global_load_b128 v[32:35], v2, s[10:11] offset:224         // 000000001910: DC5E00E0 200A0002
	global_load_b128 v[36:39], v2, s[10:11] offset:240         // 000000001918: DC5E00F0 240A0002
	v_add_nc_u32_e32 v2, v4, v6                                // 000000001920: 4A040D04
	s_waitcnt vmcnt(2) lgkmcnt(2)                              // 000000001924: BF890827
	v_wmma_f32_16x16x16_f16 v[64:71], v[24:31], v[16:23], v[64:71]// 000000001928: CC404040 1D022118
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000001930: BF890007
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001934: BF870001
	v_wmma_f32_16x16x16_f16 v[64:71], v[32:39], v[48:55], v[64:71]// 000000001938: CC404040 1D026120
	ds_store_b32 v2, v64 offset:4352                           // 000000001940: D8341100 00004002
	ds_store_b32 v2, v65 offset:4736                           // 000000001948: D8341280 00004102
	ds_store_b32 v2, v66 offset:5120                           // 000000001950: D8341400 00004202
	ds_store_b32 v2, v67 offset:5504                           // 000000001958: D8341580 00004302
	s_waitcnt lgkmcnt(0)                                       // 000000001960: BF89FC07
	s_barrier                                                  // 000000001964: BFBD0000
	v_mad_u32_u24 v2, v11, lit(0x6), v10                       // 000000001968: D60B0002 0429FF0B 00000006
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001974: BF870001
	v_lshlrev_b32_e32 v2, 4, v2                                // 000000001978: 30040484
	ds_load_b32 v4, v2 offset:4352                             // 00000000197C: D8D81100 04000002
	ds_load_b32 v5, v2 offset:4356                             // 000000001984: D8D81104 05000002
	ds_load_b32 v6, v2 offset:4360                             // 00000000198C: D8D81108 06000002
	ds_load_b32 v2, v2 offset:4364                             // 000000001994: D8D8110C 02000002
	s_waitcnt lgkmcnt(3)                                       // 00000000199C: BF89FC37
	v_add_f32_e32 v4, v4, v7                                   // 0000000019A0: 06080F04
	s_waitcnt lgkmcnt(2)                                       // 0000000019A4: BF89FC27
	v_add_f32_e32 v5, v5, v7                                   // 0000000019A8: 060A0F05
	s_waitcnt lgkmcnt(1)                                       // 0000000019AC: BF89FC17
	v_add_f32_e32 v6, v6, v7                                   // 0000000019B0: 060C0F06
	s_waitcnt lgkmcnt(0)                                       // 0000000019B4: BF89FC07
	v_add_f32_e32 v2, v2, v7                                   // 0000000019B8: 06040F02
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000019BC: BF870214
	v_cndmask_b32_e64 v4, 0xf149f2ca, v4, s24                  // 0000000019C0: D5010004 006208FF F149F2CA
	v_cndmask_b32_e64 v5, 0xf149f2ca, v5, s28                  // 0000000019CC: D5010005 00720AFF F149F2CA
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000019D8: BF870214
	v_cndmask_b32_e64 v6, 0xf149f2ca, v6, s30                  // 0000000019DC: D5010006 007A0CFF F149F2CA
	v_cndmask_b32_e64 v2, 0xf149f2ca, v2, s32                  // 0000000019E8: D5010002 008204FF F149F2CA
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019F4: BF870094
	v_max_f32_dpp v7, v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000019F8: 200E08FA FF08B104
	v_max_f32_dpp v7, v7, v7 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A00: 200E0EFA FF084E07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A08: BF870091
	v_max_f32_dpp v7, v7, v7 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A0C: 200E0EFA FF094107
	v_max_f32_dpp v7, v7, v7 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A14: 200E0EFA FF094007
	v_max_f32_dpp v8, v5, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A1C: 20100AFA FF08B105
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A24: BF870091
	v_max_f32_dpp v8, v8, v8 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A28: 201010FA FF084E08
	v_max_f32_dpp v8, v8, v8 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A30: 201010FA FF094108
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001A38: BF8700A1
	v_max_f32_dpp v8, v8, v8 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A3C: 201010FA FF094008
	v_max_f32_dpp v9, v6, v6 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A44: 20120CFA FF08B106
	v_max_f32_dpp v9, v9, v9 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A4C: 201212FA FF084E09
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A54: BF870091
	v_max_f32_dpp v9, v9, v9 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A58: 201212FA FF094109
	v_max_f32_dpp v9, v9, v9 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A60: 201212FA FF094009
	v_max_f32_dpp v12, v2, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A68: 201804FA FF08B102
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A70: BF870091
	v_max_f32_dpp v12, v12, v12 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A74: 201818FA FF084E0C
	v_max_f32_dpp v12, v12, v12 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A7C: 201818FA FF09410C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001A84: BF8700A1
	v_max_f32_dpp v12, v12, v12 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A88: 201818FA FF09400C
	v_permlanex16_b32 v13, v7, 0, 0                            // 000000001A90: D65C000D 02010107
	v_max_f32_e32 v7, v7, v13                                  // 000000001A98: 200E1B07
	v_permlanex16_b32 v13, v8, 0, 0                            // 000000001A9C: D65C000D 02010108
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001AA4: BF8700A1
	v_max_f32_e32 v8, v8, v13                                  // 000000001AA8: 20101B08
	v_permlanex16_b32 v13, v9, 0, 0                            // 000000001AAC: D65C000D 02010109
	v_max_f32_e32 v9, v9, v13                                  // 000000001AB4: 20121B09
	v_permlanex16_b32 v13, v12, 0, 0                           // 000000001AB8: D65C000D 0201010C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001AC0: BF8700A1
	v_max_f32_e32 v12, v12, v13                                // 000000001AC4: 20181B0C
	v_readlane_b32 s8, v7, 0                                   // 000000001AC8: D7600008 00010107
	v_max_f32_e32 v7, s8, v7                                   // 000000001AD0: 200E0E08
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AD4: BF870091
	v_readlane_b32 s8, v7, 63                                  // 000000001AD8: D7600008 00017F07
	v_mov_b32_e32 v7, s8                                       // 000000001AE0: 7E0E0208
	v_readlane_b32 s8, v8, 0                                   // 000000001AE4: D7600008 00010108
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AEC: BF870091
	v_max_f32_e32 v8, s8, v8                                   // 000000001AF0: 20101008
	v_readlane_b32 s8, v8, 63                                  // 000000001AF4: D7600008 00017F08
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001AFC: BF8700A1
	v_mov_b32_e32 v8, s8                                       // 000000001B00: 7E100208
	v_readlane_b32 s8, v9, 0                                   // 000000001B04: D7600008 00010109
	v_max_f32_e32 v9, s8, v9                                   // 000000001B0C: 20121208
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B10: BF870091
	v_readlane_b32 s8, v9, 63                                  // 000000001B14: D7600008 00017F09
	v_mov_b32_e32 v9, s8                                       // 000000001B1C: 7E120208
	v_readlane_b32 s8, v12, 0                                  // 000000001B20: D7600008 0001010C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B28: BF870091
	v_max_f32_e32 v12, s8, v12                                 // 000000001B2C: 20181808
	v_readlane_b32 s8, v12, 63                                 // 000000001B30: D7600008 00017F0C
	v_sub_f32_e32 v4, v4, v7                                   // 000000001B38: 08080F04
	v_sub_f32_e32 v5, v5, v8                                   // 000000001B3C: 080A1105
	v_sub_f32_e32 v6, v6, v9                                   // 000000001B40: 080C1306
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001B44: BF870214
	v_mov_b32_e32 v12, s8                                      // 000000001B48: 7E180208
	v_mul_f32_e32 v4, 0x3fb8aa3b, v4                           // 000000001B4C: 100808FF 3FB8AA3B
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001B54: BF870214
	v_mul_f32_e32 v5, 0x3fb8aa3b, v5                           // 000000001B58: 100A0AFF 3FB8AA3B
	v_mul_f32_e32 v6, 0x3fb8aa3b, v6                           // 000000001B60: 100C0CFF 3FB8AA3B
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001B68: BF8700C4
	v_sub_f32_e32 v2, v2, v12                                  // 000000001B6C: 08041902
	v_exp_f32_e32 v4, v4                                       // 000000001B70: 7E084B04
	v_exp_f32_e32 v5, v5                                       // 000000001B74: 7E0A4B05
	v_exp_f32_e32 v6, v6                                       // 000000001B78: 7E0C4B06
	v_mul_f32_e32 v2, 0x3fb8aa3b, v2                           // 000000001B7C: 100404FF 3FB8AA3B
	v_cndmask_b32_e64 v4, 0, v4, s24                           // 000000001B84: D5010004 00620880
	s_waitcnt_depctr depctr_hold_cnt(0) depctr_sa_sdst(0) depctr_va_vdst(0) depctr_va_sdst(0) depctr_va_ssrc(0) depctr_va_vcc(0) depctr_vm_vsrc(0)// 000000001B8C: BF880000
	v_cndmask_b32_e64 v5, 0, v5, s28                           // 000000001B90: D5010005 00720A80
	v_cndmask_b32_e64 v6, 0, v6, s30                           // 000000001B98: D5010006 007A0C80
	v_exp_f32_e32 v2, v2                                       // 000000001BA0: 7E044B02
	s_waitcnt_depctr depctr_hold_cnt(0) depctr_sa_sdst(0) depctr_va_vdst(0) depctr_va_sdst(0) depctr_va_ssrc(0) depctr_va_vcc(0) depctr_vm_vsrc(0)// 000000001BA4: BF880000
	v_cndmask_b32_e64 v2, 0, v2, s32                           // 000000001BA8: D5010002 00820480
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BB0: BF870094
	v_add_f32_dpp v13, v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001BB4: 061A08FA FF08B104
	v_add_f32_dpp v13, v13, v13 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001BBC: 061A1AFA FF084E0D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BC4: BF870091
	v_add_f32_dpp v13, v13, v13 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001BC8: 061A1AFA FF09410D
	v_add_f32_dpp v13, v13, v13 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001BD0: 061A1AFA FF09400D
	v_add_f32_dpp v14, v5, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001BD8: 061C0AFA FF08B105
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BE0: BF870091
	v_add_f32_dpp v14, v14, v14 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001BE4: 061C1CFA FF084E0E
	v_add_f32_dpp v14, v14, v14 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001BEC: 061C1CFA FF09410E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001BF4: BF8700A1
	v_add_f32_dpp v14, v14, v14 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001BF8: 061C1CFA FF09400E
	v_add_f32_dpp v15, v6, v6 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001C00: 061E0CFA FF08B106
	v_add_f32_dpp v15, v15, v15 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001C08: 061E1EFA FF084E0F
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C10: BF870091
	v_add_f32_dpp v15, v15, v15 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001C14: 061E1EFA FF09410F
	v_add_f32_dpp v15, v15, v15 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001C1C: 061E1EFA FF09400F
	v_add_f32_dpp v16, v2, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001C24: 062004FA FF08B102
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C2C: BF870091
	v_add_f32_dpp v16, v16, v16 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001C30: 062020FA FF084E10
	v_add_f32_dpp v16, v16, v16 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001C38: 062020FA FF094110
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001C40: BF8700A1
	v_add_f32_dpp v16, v16, v16 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001C44: 062020FA FF094010
	v_permlanex16_b32 v17, v13, 0, 0                           // 000000001C4C: D65C0011 0201010D
	v_add_f32_e32 v13, v13, v17                                // 000000001C54: 061A230D
	v_permlanex16_b32 v17, v14, 0, 0                           // 000000001C58: D65C0011 0201010E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001C60: BF8700A1
	v_add_f32_e32 v14, v14, v17                                // 000000001C64: 061C230E
	v_permlanex16_b32 v17, v15, 0, 0                           // 000000001C68: D65C0011 0201010F
	v_add_f32_e32 v15, v15, v17                                // 000000001C70: 061E230F
	v_permlanex16_b32 v17, v16, 0, 0                           // 000000001C74: D65C0011 02010110
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001C7C: BF8700A1
	v_add_f32_e32 v16, v16, v17                                // 000000001C80: 06202310
	v_readlane_b32 s8, v13, 0                                  // 000000001C84: D7600008 0001010D
	v_add_f32_e32 v13, s8, v13                                 // 000000001C8C: 061A1A08
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C90: BF870091
	v_readlane_b32 s8, v13, 63                                 // 000000001C94: D7600008 00017F0D
	v_mov_b32_e32 v13, s8                                      // 000000001C9C: 7E1A0208
	v_readlane_b32 s8, v14, 0                                  // 000000001CA0: D7600008 0001010E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CA8: BF870091
	v_add_f32_e32 v14, s8, v14                                 // 000000001CAC: 061C1C08
	v_readlane_b32 s8, v14, 63                                 // 000000001CB0: D7600008 00017F0E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001CB8: BF8700A1
	v_mov_b32_e32 v14, s8                                      // 000000001CBC: 7E1C0208
	v_readlane_b32 s8, v15, 0                                  // 000000001CC0: D7600008 0001010F
	v_add_f32_e32 v15, s8, v15                                 // 000000001CC8: 061E1E08
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CCC: BF870091
	v_readlane_b32 s8, v15, 63                                 // 000000001CD0: D7600008 00017F0F
	v_mov_b32_e32 v15, s8                                      // 000000001CD8: 7E1E0208
	v_readlane_b32 s8, v16, 0                                  // 000000001CDC: D7600008 00010110
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CE4: BF870091
	v_add_f32_e32 v16, s8, v16                                 // 000000001CE8: 06202008
	v_readlane_b32 s8, v16, 63                                 // 000000001CEC: D7600008 00017F10
	v_cvt_f16_f32_e32 v4.l, v4                                 // 000000001CF4: 7E081504
	v_cvt_f16_f32_e32 v5.l, v5                                 // 000000001CF8: 7E0A1505
	v_mad_u32_u24 v16, v11, lit(0x6), v10                      // 000000001CFC: D60B0010 0429FF0B 00000006
	v_cvt_f16_f32_e32 v6.l, v6                                 // 000000001D08: 7E0C1506
	v_mov_b32_e32 v17, s8                                      // 000000001D0C: 7E220208
	v_bfe_u32 v4, v4, 0, 16                                    // 000000001D10: D6100004 02410104
	v_bfe_u32 v5, v5, 0, 16                                    // 000000001D18: D6100005 02410105
	v_lshlrev_b32_e32 v16, 3, v16                              // 000000001D20: 30202083
	v_cvt_f16_f32_e32 v2.l, v2                                 // 000000001D24: 7E041502
	v_bfe_u32 v6, v6, 0, 16                                    // 000000001D28: D6100006 02410106
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001D30: BF870194
	v_lshl_add_u32 v4, v5, 16, v4                              // 000000001D34: D6460004 04112105
	v_bfe_u32 v2, v2, 0, 16                                    // 000000001D3C: D6100002 02410102
	ds_store_b16 v16, v4 offset:10496                          // 000000001D44: D87C2900 00000410
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001D4C: BF870002
	v_lshrrev_b32_e32 v4, 16, v4                               // 000000001D50: 32080890
	s_delay_alu instid0(VALU_DEP_2) | instid1(VALU_DEP_4)      // 000000001D54: BF870202
	v_lshl_add_u32 v2, v2, 16, v6                              // 000000001D58: D6460002 04192102
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001D60: BF870001
	v_lshrrev_b32_e32 v5, 16, v2                               // 000000001D64: 320A0490
	ds_store_b16 v16, v4 offset:10498                          // 000000001D68: D87C2902 00000410
	ds_store_b16 v16, v2 offset:10500                          // 000000001D70: D87C2904 00000210
	ds_store_b16 v16, v5 offset:10502                          // 000000001D78: D87C2906 00000510
	s_waitcnt lgkmcnt(0)                                       // 000000001D80: BF89FC07
	s_barrier                                                  // 000000001D84: BFBD0000
	v_lshlrev_b32_e32 v2, 6, v10                               // 000000001D88: 30041486
	v_lshlrev_b32_e32 v3, 1, v3                                // 000000001D8C: 30060681
	s_lshl_b32 s8, s2, 16                                      // 000000001D90: 84089002
	s_lshl_b32 s9, s3, 8                                       // 000000001D94: 84098803
	s_add_u32 s8, s8, s9                                       // 000000001D98: 80080908
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001D9C: BF870002
	v_add_nc_u32_e32 v4, s8, v2                                // 000000001DA0: 4A080408
	ds_load_u16_d16 v24, v3 offset:10496                       // 000000001DA4: DA982900 18000003
	ds_load_u16_d16 v25, v3 offset:10592                       // 000000001DAC: DA982960 19000003
	ds_load_u16_d16 v26, v3 offset:10688                       // 000000001DB4: DA9829C0 1A000003
	ds_load_u16_d16 v27, v3 offset:10784                       // 000000001DBC: DA982A20 1B000003
	ds_load_u16_d16 v28, v3 offset:10880                       // 000000001DC4: DA982A80 1C000003
	ds_load_u16_d16 v29, v3 offset:10976                       // 000000001DCC: DA982AE0 1D000003
	ds_load_u16_d16 v30, v3 offset:11072                       // 000000001DD4: DA982B40 1E000003
	ds_load_u16_d16 v31, v3 offset:11168                       // 000000001DDC: DA982BA0 1F000003
	ds_load_u16_d16_hi v24, v3 offset:10544                    // 000000001DE4: DA9C2930 18000003
	ds_load_u16_d16_hi v25, v3 offset:10640                    // 000000001DEC: DA9C2990 19000003
	ds_load_u16_d16_hi v26, v3 offset:10736                    // 000000001DF4: DA9C29F0 1A000003
	ds_load_u16_d16_hi v27, v3 offset:10832                    // 000000001DFC: DA9C2A50 1B000003
	ds_load_u16_d16_hi v28, v3 offset:10928                    // 000000001E04: DA9C2AB0 1C000003
	ds_load_u16_d16_hi v29, v3 offset:11024                    // 000000001E0C: DA9C2B10 1D000003
	ds_load_u16_d16_hi v30, v3 offset:11120                    // 000000001E14: DA9C2B70 1E000003
	ds_load_u16_d16_hi v31, v3 offset:11216                    // 000000001E1C: DA9C2BD0 1F000003
	ds_load_u16_d16 v32, v3 offset:11264                       // 000000001E24: DA982C00 20000003
	ds_load_u16_d16 v33, v3 offset:11360                       // 000000001E2C: DA982C60 21000003
	ds_load_u16_d16 v34, v3 offset:11456                       // 000000001E34: DA982CC0 22000003
	ds_load_u16_d16 v35, v3 offset:11552                       // 000000001E3C: DA982D20 23000003
	ds_load_u16_d16 v36, v3 offset:11648                       // 000000001E44: DA982D80 24000003
	ds_load_u16_d16 v37, v3 offset:11744                       // 000000001E4C: DA982DE0 25000003
	ds_load_u16_d16 v38, v3 offset:11840                       // 000000001E54: DA982E40 26000003
	ds_load_u16_d16 v39, v3 offset:11936                       // 000000001E5C: DA982EA0 27000003
	ds_load_u16_d16_hi v32, v3 offset:11312                    // 000000001E64: DA9C2C30 20000003
	ds_load_u16_d16_hi v33, v3 offset:11408                    // 000000001E6C: DA9C2C90 21000003
	ds_load_u16_d16_hi v34, v3 offset:11504                    // 000000001E74: DA9C2CF0 22000003
	ds_load_u16_d16_hi v35, v3 offset:11600                    // 000000001E7C: DA9C2D50 23000003
	ds_load_u16_d16_hi v36, v3 offset:11696                    // 000000001E84: DA9C2DB0 24000003
	ds_load_u16_d16_hi v37, v3 offset:11792                    // 000000001E8C: DA9C2E10 25000003
	ds_load_u16_d16_hi v38, v3 offset:11888                    // 000000001E94: DA9C2E70 26000003
	ds_load_u16_d16_hi v39, v3 offset:11984                    // 000000001E9C: DA9C2ED0 27000003
	ds_load_u16_d16 v40, v3 offset:12032                       // 000000001EA4: DA982F00 28000003
	ds_load_u16_d16 v41, v3 offset:12128                       // 000000001EAC: DA982F60 29000003
	ds_load_u16_d16 v42, v3 offset:12224                       // 000000001EB4: DA982FC0 2A000003
	ds_load_u16_d16 v43, v3 offset:12320                       // 000000001EBC: DA983020 2B000003
	ds_load_u16_d16 v44, v3 offset:12416                       // 000000001EC4: DA983080 2C000003
	ds_load_u16_d16 v45, v3 offset:12512                       // 000000001ECC: DA9830E0 2D000003
	ds_load_u16_d16 v46, v3 offset:12608                       // 000000001ED4: DA983140 2E000003
	ds_load_u16_d16 v47, v3 offset:12704                       // 000000001EDC: DA9831A0 2F000003
	ds_load_u16_d16_hi v40, v3 offset:12080                    // 000000001EE4: DA9C2F30 28000003
	ds_load_u16_d16_hi v41, v3 offset:12176                    // 000000001EEC: DA9C2F90 29000003
	ds_load_u16_d16_hi v42, v3 offset:12272                    // 000000001EF4: DA9C2FF0 2A000003
	ds_load_u16_d16_hi v43, v3 offset:12368                    // 000000001EFC: DA9C3050 2B000003
	ds_load_u16_d16_hi v44, v3 offset:12464                    // 000000001F04: DA9C30B0 2C000003
	ds_load_u16_d16_hi v45, v3 offset:12560                    // 000000001F0C: DA9C3110 2D000003
	ds_load_u16_d16_hi v46, v3 offset:12656                    // 000000001F14: DA9C3170 2E000003
	ds_load_u16_d16_hi v47, v3 offset:12752                    // 000000001F1C: DA9C31D0 2F000003
	ds_load_u16_d16 v48, v3 offset:12800                       // 000000001F24: DA983200 30000003
	ds_load_u16_d16 v49, v3 offset:12896                       // 000000001F2C: DA983260 31000003
	ds_load_u16_d16 v50, v3 offset:12992                       // 000000001F34: DA9832C0 32000003
	ds_load_u16_d16 v51, v3 offset:13088                       // 000000001F3C: DA983320 33000003
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_2)      // 000000001F44: BF870101
	v_add_nc_u32_e32 v4, v4, v3                                // 000000001F48: 4A080704
	ds_load_u16_d16 v52, v3 offset:13184                       // 000000001F4C: DA983380 34000003
	ds_load_u16_d16 v53, v3 offset:13280                       // 000000001F54: DA9833E0 35000003
	s_lshl_b32 s8, s2, 16                                      // 000000001F5C: 84089002
	ds_load_u16_d16_hi v48, v3 offset:12848                    // 000000001F60: DA9C3230 30000003
	global_load_d16_b16 v56, v4, s[12:13]                      // 000000001F68: DC820000 380C0004
	global_load_d16_b16 v57, v4, s[12:13] offset:2048          // 000000001F70: DC820800 390C0004
	global_load_d16_b16 v64, v4, s[12:13] offset:32            // 000000001F78: DC820020 400C0004
	global_load_d16_b16 v65, v4, s[12:13] offset:2080          // 000000001F80: DC820820 410C0004
	ds_load_u16_d16 v54, v3 offset:13376                       // 000000001F88: DA983440 36000003
	ds_load_u16_d16_hi v49, v3 offset:12944                    // 000000001F90: DA9C3290 31000003
	ds_load_u16_d16 v55, v3 offset:13472                       // 000000001F98: DA9834A0 37000003
	ds_load_u16_d16_hi v50, v3 offset:13040                    // 000000001FA0: DA9C32F0 32000003
	ds_load_u16_d16_hi v51, v3 offset:13136                    // 000000001FA8: DA9C3350 33000003
	ds_load_u16_d16_hi v52, v3 offset:13232                    // 000000001FB0: DA9C33B0 34000003
	ds_load_u16_d16_hi v53, v3 offset:13328                    // 000000001FB8: DA9C3410 35000003
	v_add_nc_u32_e32 v5, 0x1000, v4                            // 000000001FC0: 4A0A08FF 00001000
	ds_load_u16_d16_hi v54, v3 offset:13424                    // 000000001FC8: DA9C3470 36000003
	v_add_nc_u32_e32 v6, 0x2000, v4                            // 000000001FD0: 4A0C08FF 00002000
	v_add_nc_u32_e32 v16, 0x3000, v4                           // 000000001FD8: 4A2008FF 00003000
	v_lshlrev_b32_e32 v18, lit(0x2), v11                       // 000000001FE0: 302416FF 00000002
	global_load_d16_b16 v58, v5, s[12:13]                      // 000000001FE8: DC820000 3A0C0005
	global_load_d16_b16 v59, v5, s[12:13] offset:2048          // 000000001FF0: DC820800 3B0C0005
	global_load_d16_b16 v60, v6, s[12:13]                      // 000000001FF8: DC820000 3C0C0006
	global_load_d16_b16 v61, v6, s[12:13] offset:2048          // 000000002000: DC820800 3D0C0006
	global_load_d16_b16 v62, v16, s[12:13]                     // 000000002008: DC820000 3E0C0010
	global_load_d16_b16 v63, v16, s[12:13] offset:2048         // 000000002010: DC820800 3F0C0010
	global_load_d16_hi_b16 v56, v4, s[12:13] offset:1024       // 000000002018: DC8E0400 380C0004
	global_load_d16_hi_b16 v57, v4, s[12:13] offset:3072       // 000000002020: DC8E0C00 390C0004
	global_load_d16_b16 v66, v5, s[12:13] offset:32            // 000000002028: DC820020 420C0005
	global_load_d16_b16 v67, v5, s[12:13] offset:2080          // 000000002030: DC820820 430C0005
	global_load_d16_hi_b16 v58, v5, s[12:13] offset:1024       // 000000002038: DC8E0400 3A0C0005
	global_load_d16_hi_b16 v59, v5, s[12:13] offset:3072       // 000000002040: DC8E0C00 3B0C0005
	global_load_d16_hi_b16 v60, v6, s[12:13] offset:1024       // 000000002048: DC8E0400 3C0C0006
	global_load_d16_hi_b16 v61, v6, s[12:13] offset:3072       // 000000002050: DC8E0C00 3D0C0006
	global_load_d16_hi_b16 v62, v16, s[12:13] offset:1024      // 000000002058: DC8E0400 3E0C0010
	global_load_d16_hi_b16 v63, v16, s[12:13] offset:3072      // 000000002060: DC8E0C00 3F0C0010
	s_waitcnt vmcnt(0) lgkmcnt(47)                             // 000000002068: BF8902F7
	v_wmma_f16_16x16x16_f16 v[20:27], v[24:31], v[56:63], 0    // 00000000206C: CC424014 1A027118
	global_load_d16_hi_b16 v64, v4, s[12:13] offset:1056       // 000000002074: DC8E0420 400C0004
	global_load_d16_hi_b16 v65, v4, s[12:13] offset:3104       // 00000000207C: DC8E0C20 410C0004
	global_load_d16_hi_b16 v66, v5, s[12:13] offset:1056       // 000000002084: DC8E0420 420C0005
	global_load_d16_hi_b16 v67, v5, s[12:13] offset:3104       // 00000000208C: DC8E0C20 430C0005
	ds_load_u16_d16_hi v55, v3 offset:13520                    // 000000002094: DA9C34D0 37000003
	global_load_d16_b16 v68, v6, s[12:13] offset:32            // 00000000209C: DC820020 440C0006
	global_load_d16_b16 v69, v6, s[12:13] offset:2080          // 0000000020A4: DC820820 450C0006
	global_load_d16_b16 v70, v16, s[12:13] offset:32           // 0000000020AC: DC820020 460C0010
	global_load_d16_b16 v71, v16, s[12:13] offset:2080         // 0000000020B4: DC820820 470C0010
	s_lshl_b32 s9, s3, 8                                       // 0000000020BC: 84098803
	s_add_u32 s8, s8, s9                                       // 0000000020C0: 80080908
	v_add_nc_u32_e32 v4, s8, v2                                // 0000000020C4: 4A080408
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000020C8: BF870001
	v_add_nc_u32_e32 v4, v4, v3                                // 0000000020CC: 4A080704
	s_lshl_b32 s8, s2, 16                                      // 0000000020D0: 84089002
	s_lshl_b32 s9, s3, 8                                       // 0000000020D4: 84098803
	s_add_u32 s8, s8, s9                                       // 0000000020D8: 80080908
	v_add_nc_u32_e32 v5, s8, v2                                // 0000000020DC: 4A0A0408
	global_load_d16_hi_b16 v68, v6, s[12:13] offset:1056       // 0000000020E0: DC8E0420 440C0006
	global_load_d16_hi_b16 v69, v6, s[12:13] offset:3104       // 0000000020E8: DC8E0C20 450C0006
	global_load_d16_hi_b16 v70, v16, s[12:13] offset:1056      // 0000000020F0: DC8E0420 460C0010
	global_load_d16_hi_b16 v71, v16, s[12:13] offset:3104      // 0000000020F8: DC8E0C20 470C0010
	s_waitcnt vmcnt(0)                                         // 000000002100: BF8903F7
	v_wmma_f16_16x16x16_f16 v[56:63], v[24:31], v[64:71], 0    // 000000002104: CC424038 1A028118
	v_add_nc_u32_e32 v5, v5, v3                                // 00000000210C: 4A0A0705
	v_add_nc_u32_e32 v6, 0x4000, v4                            // 000000002110: 4A0C08FF 00004000
	v_add_nc_u32_e32 v16, 0x5000, v4                           // 000000002118: 4A2008FF 00005000
	v_add_nc_u32_e32 v19, 0x6000, v4                           // 000000002120: 4A2608FF 00006000
	v_add_nc_u32_e32 v4, 0x7000, v4                            // 000000002128: 4A0808FF 00007000
	v_mov_b32_e32 v24, 0x80                                    // 000000002130: 7E3002FF 00000080
	global_load_d16_b16 v64, v6, s[12:13]                      // 000000002138: DC820000 400C0006
	global_load_d16_b16 v65, v6, s[12:13] offset:2048          // 000000002140: DC820800 410C0006
	global_load_d16_b16 v66, v16, s[12:13]                     // 000000002148: DC820000 420C0010
	global_load_d16_b16 v67, v16, s[12:13] offset:2048         // 000000002150: DC820800 430C0010
	global_load_d16_b16 v68, v19, s[12:13]                     // 000000002158: DC820000 440C0013
	global_load_d16_b16 v69, v19, s[12:13] offset:2048         // 000000002160: DC820800 450C0013
	global_load_d16_b16 v70, v4, s[12:13]                      // 000000002168: DC820000 460C0004
	global_load_d16_b16 v71, v4, s[12:13] offset:2048          // 000000002170: DC820800 470C0004
	global_load_d16_b16 v72, v6, s[12:13] offset:32            // 000000002178: DC820020 480C0006
	global_load_d16_b16 v73, v6, s[12:13] offset:2080          // 000000002180: DC820820 490C0006
	global_load_d16_hi_b16 v64, v6, s[12:13] offset:1024       // 000000002188: DC8E0400 400C0006
	global_load_d16_hi_b16 v65, v6, s[12:13] offset:3072       // 000000002190: DC8E0C00 410C0006
	global_load_d16_hi_b16 v66, v16, s[12:13] offset:1024      // 000000002198: DC8E0400 420C0010
	global_load_d16_hi_b16 v67, v16, s[12:13] offset:3072      // 0000000021A0: DC8E0C00 430C0010
	global_load_d16_hi_b16 v68, v19, s[12:13] offset:1024      // 0000000021A8: DC8E0400 440C0013
	global_load_d16_hi_b16 v69, v19, s[12:13] offset:3072      // 0000000021B0: DC8E0C00 450C0013
	global_load_d16_hi_b16 v70, v4, s[12:13] offset:1024       // 0000000021B8: DC8E0400 460C0004
	global_load_d16_hi_b16 v71, v4, s[12:13] offset:3072       // 0000000021C0: DC8E0C00 470C0004
	s_waitcnt vmcnt(0) lgkmcnt(32)                             // 0000000021C8: BF890207
	v_wmma_f16_16x16x16_f16 v[20:27], v[32:39], v[64:71], v[20:27]// 0000000021CC: CC424014 1C528120
	global_load_d16_b16 v74, v16, s[12:13] offset:32           // 0000000021D4: DC820020 4A0C0010
	global_load_d16_b16 v75, v16, s[12:13] offset:2080         // 0000000021DC: DC820820 4B0C0010
	global_load_d16_b16 v76, v19, s[12:13] offset:32           // 0000000021E4: DC820020 4C0C0013
	global_load_d16_b16 v77, v19, s[12:13] offset:2080         // 0000000021EC: DC820820 4D0C0013
	global_load_d16_b16 v78, v4, s[12:13] offset:32            // 0000000021F4: DC820020 4E0C0004
	global_load_d16_b16 v79, v4, s[12:13] offset:2080          // 0000000021FC: DC820820 4F0C0004
	global_load_d16_hi_b16 v72, v6, s[12:13] offset:1056       // 000000002204: DC8E0420 480C0006
	global_load_d16_hi_b16 v73, v6, s[12:13] offset:3104       // 00000000220C: DC8E0C20 490C0006
	v_add_nc_u32_e32 v6, 0x8000, v5                            // 000000002214: 4A0C0AFF 00008000
	v_add_nc_u32_e32 v25, 0x9000, v5                           // 00000000221C: 4A320AFF 00009000
	v_add_nc_u32_e32 v26, 0xa000, v5                           // 000000002224: 4A340AFF 0000A000
	v_add_nc_u32_e32 v5, 0xb000, v5                            // 00000000222C: 4A0A0AFF 0000B000
	s_lshl_b32 s8, s2, 16                                      // 000000002234: 84089002
	s_lshl_b32 s9, s3, 8                                       // 000000002238: 84098803
	s_add_u32 s8, s8, s9                                       // 00000000223C: 80080908
	global_load_d16_hi_b16 v74, v16, s[12:13] offset:1056      // 000000002240: DC8E0420 4A0C0010
	global_load_d16_hi_b16 v75, v16, s[12:13] offset:3104      // 000000002248: DC8E0C20 4B0C0010
	global_load_d16_b16 v64, v6, s[12:13]                      // 000000002250: DC820000 400C0006
	global_load_d16_hi_b16 v76, v19, s[12:13] offset:1056      // 000000002258: DC8E0420 4C0C0013
	global_load_d16_hi_b16 v77, v19, s[12:13] offset:3104      // 000000002260: DC8E0C20 4D0C0013
	global_load_d16_b16 v65, v6, s[12:13] offset:2048          // 000000002268: DC820800 410C0006
	global_load_d16_hi_b16 v78, v4, s[12:13] offset:1056       // 000000002270: DC8E0420 4E0C0004
	global_load_d16_hi_b16 v79, v4, s[12:13] offset:3104       // 000000002278: DC8E0C20 4F0C0004
	global_load_d16_b16 v66, v25, s[12:13]                     // 000000002280: DC820000 420C0019
	s_waitcnt vmcnt(1)                                         // 000000002288: BF8907F7
	v_wmma_f16_16x16x16_f16 v[56:63], v[32:39], v[72:79], v[56:63]// 00000000228C: CC424038 1CE29120
	global_load_d16_b16 v67, v25, s[12:13] offset:2048         // 000000002294: DC820800 430C0019
	global_load_d16_b16 v68, v26, s[12:13]                     // 00000000229C: DC820000 440C001A
	global_load_d16_b16 v69, v26, s[12:13] offset:2048         // 0000000022A4: DC820800 450C001A
	global_load_d16_b16 v70, v5, s[12:13]                      // 0000000022AC: DC820000 460C0005
	global_load_d16_b16 v71, v5, s[12:13] offset:2048          // 0000000022B4: DC820800 470C0005
	global_load_d16_hi_b16 v64, v6, s[12:13] offset:1024       // 0000000022BC: DC8E0400 400C0006
	global_load_d16_hi_b16 v65, v6, s[12:13] offset:3072       // 0000000022C4: DC8E0C00 410C0006
	global_load_d16_hi_b16 v66, v25, s[12:13] offset:1024      // 0000000022CC: DC8E0400 420C0019
	global_load_d16_b16 v32, v6, s[12:13] offset:32            // 0000000022D4: DC820020 200C0006
	global_load_d16_b16 v33, v6, s[12:13] offset:2080          // 0000000022DC: DC820820 210C0006
	global_load_d16_b16 v34, v25, s[12:13] offset:32           // 0000000022E4: DC820020 220C0019
	global_load_d16_b16 v35, v25, s[12:13] offset:2080         // 0000000022EC: DC820820 230C0019
	global_load_d16_b16 v36, v26, s[12:13] offset:32           // 0000000022F4: DC820020 240C001A
	global_load_d16_b16 v37, v26, s[12:13] offset:2080         // 0000000022FC: DC820820 250C001A
	global_load_d16_b16 v38, v5, s[12:13] offset:32            // 000000002304: DC820020 260C0005
	global_load_d16_b16 v39, v5, s[12:13] offset:2080          // 00000000230C: DC820820 270C0005
	global_load_d16_hi_b16 v67, v25, s[12:13] offset:3072      // 000000002314: DC8E0C00 430C0019
	global_load_d16_hi_b16 v68, v26, s[12:13] offset:1024      // 00000000231C: DC8E0400 440C001A
	global_load_d16_hi_b16 v69, v26, s[12:13] offset:3072      // 000000002324: DC8E0C00 450C001A
	global_load_d16_hi_b16 v70, v5, s[12:13] offset:1024       // 00000000232C: DC8E0400 460C0005
	global_load_d16_hi_b16 v71, v5, s[12:13] offset:3072       // 000000002334: DC8E0C00 470C0005
	v_add_nc_u32_e32 v4, s8, v2                                // 00000000233C: 4A080408
	s_waitcnt vmcnt(0) lgkmcnt(16)                             // 000000002340: BF890107
	v_wmma_f16_16x16x16_f16 v[20:27], v[40:47], v[64:71], v[20:27]// 000000002344: CC424014 1C528128
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000234C: BF870002
	v_add_nc_u32_e32 v4, v4, v3                                // 000000002350: 4A080704
	global_load_d16_hi_b16 v32, v6, s[12:13] offset:1056       // 000000002354: DC8E0420 200C0006
	global_load_d16_hi_b16 v33, v6, s[12:13] offset:3104       // 00000000235C: DC8E0C20 210C0006
	global_load_d16_hi_b16 v34, v25, s[12:13] offset:1056      // 000000002364: DC8E0420 220C0019
	global_load_d16_hi_b16 v35, v25, s[12:13] offset:3104      // 00000000236C: DC8E0C20 230C0019
	global_load_d16_hi_b16 v36, v26, s[12:13] offset:1056      // 000000002374: DC8E0420 240C001A
	global_load_d16_hi_b16 v37, v26, s[12:13] offset:3104      // 00000000237C: DC8E0C20 250C001A
	global_load_d16_hi_b16 v38, v5, s[12:13] offset:1056       // 000000002384: DC8E0420 260C0005
	global_load_d16_hi_b16 v39, v5, s[12:13] offset:3104       // 00000000238C: DC8E0C20 270C0005
	v_add_nc_u32_e32 v2, v2, v3                                // 000000002394: 4A040702
	v_add_nc_u32_e32 v3, 0xc000, v4                            // 000000002398: 4A0608FF 0000C000
	v_add_nc_u32_e32 v5, 0xd000, v4                            // 0000000023A0: 4A0A08FF 0000D000
	v_add_nc_u32_e32 v6, 0xe000, v4                            // 0000000023A8: 4A0C08FF 0000E000
	v_add_nc_u32_e32 v4, 0xf000, v4                            // 0000000023B0: 4A0808FF 0000F000
	v_cmp_lt_u32_e64 s8, v18, v24                              // 0000000023B8: D4490008 02023112
	global_load_d16_b16 v24, v3, s[12:13]                      // 0000000023C0: DC820000 180C0003
	global_load_d16_b16 v25, v3, s[12:13] offset:2048          // 0000000023C8: DC820800 190C0003
	global_load_d16_b16 v26, v5, s[12:13]                      // 0000000023D0: DC820000 1A0C0005
	global_load_d16_b16 v27, v5, s[12:13] offset:2048          // 0000000023D8: DC820800 1B0C0005
	global_load_d16_b16 v28, v6, s[12:13]                      // 0000000023E0: DC820000 1C0C0006
	global_load_d16_b16 v29, v6, s[12:13] offset:2048          // 0000000023E8: DC820800 1D0C0006
	global_load_d16_b16 v30, v4, s[12:13]                      // 0000000023F0: DC820000 1E0C0004
	global_load_d16_b16 v31, v4, s[12:13] offset:2048          // 0000000023F8: DC820800 1F0C0004
	global_load_d16_b16 v64, v3, s[12:13] offset:32            // 000000002400: DC820020 400C0003
	global_load_d16_b16 v65, v3, s[12:13] offset:2080          // 000000002408: DC820820 410C0003
	global_load_d16_b16 v66, v5, s[12:13] offset:32            // 000000002410: DC820020 420C0005
	global_load_d16_b16 v67, v5, s[12:13] offset:2080          // 000000002418: DC820820 430C0005
	global_load_d16_b16 v68, v6, s[12:13] offset:32            // 000000002420: DC820020 440C0006
	s_waitcnt vmcnt(13)                                        // 000000002428: BF8937F7
	v_wmma_f16_16x16x16_f16 v[56:63], v[40:47], v[32:39], v[56:63]// 00000000242C: CC424038 1CE24128
	global_load_d16_b16 v69, v6, s[12:13] offset:2080          // 000000002434: DC820820 450C0006
	v_cndmask_b32_e64 v16, 0, v18, s8                          // 00000000243C: D5010010 00222480
	global_load_d16_hi_b16 v24, v3, s[12:13] offset:1024       // 000000002444: DC8E0400 180C0003
	global_load_d16_hi_b16 v25, v3, s[12:13] offset:3072       // 00000000244C: DC8E0C00 190C0003
	global_load_d16_hi_b16 v26, v5, s[12:13] offset:1024       // 000000002454: DC8E0400 1A0C0005
	global_load_d16_hi_b16 v27, v5, s[12:13] offset:3072       // 00000000245C: DC8E0C00 1B0C0005
	global_load_d16_hi_b16 v28, v6, s[12:13] offset:1024       // 000000002464: DC8E0400 1C0C0006
	global_load_d16_hi_b16 v29, v6, s[12:13] offset:3072       // 00000000246C: DC8E0C00 1D0C0006
	global_load_d16_hi_b16 v30, v4, s[12:13] offset:1024       // 000000002474: DC8E0400 1E0C0004
	global_load_d16_hi_b16 v31, v4, s[12:13] offset:3072       // 00000000247C: DC8E0C00 1F0C0004
	global_load_d16_b16 v70, v4, s[12:13] offset:32            // 000000002484: DC820020 460C0004
	global_load_d16_b16 v71, v4, s[12:13] offset:2080          // 00000000248C: DC820820 470C0004
	global_load_d16_hi_b16 v64, v3, s[12:13] offset:1056       // 000000002494: DC8E0420 400C0003
	global_load_d16_hi_b16 v65, v3, s[12:13] offset:3104       // 00000000249C: DC8E0C20 410C0003
	global_load_d16_hi_b16 v66, v5, s[12:13] offset:1056       // 0000000024A4: DC8E0420 420C0005
	global_load_d16_hi_b16 v67, v5, s[12:13] offset:3104       // 0000000024AC: DC8E0C20 430C0005
	global_load_d16_hi_b16 v68, v6, s[12:13] offset:1056       // 0000000024B4: DC8E0420 440C0006
	global_load_d16_hi_b16 v69, v6, s[12:13] offset:3104       // 0000000024BC: DC8E0C20 450C0006
	v_lshlrev_b32_e32 v1, 8, v1                                // 0000000024C4: 30020288
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000024C8: BF870001
	v_add_nc_u32_e32 v1, v2, v1                                // 0000000024CC: 4A020302
	s_waitcnt vmcnt(8) lgkmcnt(0)                              // 0000000024D0: BF892007
	v_wmma_f16_16x16x16_f16 v[20:27], v[48:55], v[24:31], v[20:27]// 0000000024D4: CC424014 1C523130
	global_load_d16_hi_b16 v70, v4, s[12:13] offset:1056       // 0000000024DC: DC8E0420 460C0004
	global_load_d16_hi_b16 v71, v4, s[12:13] offset:3104       // 0000000024E4: DC8E0C20 470C0004
	s_waitcnt vmcnt(0)                                         // 0000000024EC: BF8903F7
	v_wmma_f16_16x16x16_f16 v[56:63], v[48:55], v[64:71], v[56:63]// 0000000024F0: CC424038 1CE28130
	ds_store_b16 v1, v20 offset:13568                          // 0000000024F8: D87C3500 00001401
	ds_store_b16 v1, v21 offset:14592                          // 000000002500: D87C3900 00001501
	ds_store_b16 v1, v22 offset:15616                          // 000000002508: D87C3D00 00001601
	ds_store_b16 v1, v23 offset:16640                          // 000000002510: D87C4100 00001701
	ds_store_b16 v1, v56 offset:13600                          // 000000002518: D87C3520 00003801
	ds_store_b16 v1, v57 offset:14624                          // 000000002520: D87C3920 00003901
	ds_store_b16 v1, v58 offset:15648                          // 000000002528: D87C3D20 00003A01
	ds_store_b16 v1, v59 offset:16672                          // 000000002530: D87C4120 00003B01
	s_waitcnt lgkmcnt(0)                                       // 000000002538: BF89FC07
	s_barrier                                                  // 00000000253C: BFBD0000
	s_and_b64 s[0:1], s[0:1], s[8:9]                           // 000000002540: 8B800800
	s_and_saveexec_b64 s[0:1], s[0:1]                          // 000000002544: BE802100
	s_cbranch_scc0 924                                         // 000000002548: BFA1039C <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x23bc>
	v_lshl_add_u32 v1, v10, 9, v16                             // 00000000254C: D6460001 0441130A
	s_mov_b64 s[8:9], exec                                     // 000000002554: BE88017E
	s_and_b64 s[8:9], s[24:25], s[8:9]                         // 000000002558: 8B880818
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000255C: BF870001
	v_lshlrev_b32_e32 v1, 1, v1                                // 000000002560: 30020281
	ds_load_b64 v[2:3], v1 offset:13568                        // 000000002564: D9D83500 02000001
	ds_load_b64 v[4:5], v1 offset:13824                        // 00000000256C: D9D83600 04000001
	ds_load_b64 v[18:19], v1 offset:14080                      // 000000002574: D9D83700 12000001
	ds_load_b64 v[20:21], v1 offset:14336                      // 00000000257C: D9D83800 14000001
	s_cbranch_scc0 11                                          // 000000002584: BFA1000B <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x15b4>
	s_mov_b32 s8, 0x24000                                      // 000000002588: BE8800FF 00024000
	s_mul_i32 s8, s3, s8                                       // 000000002590: 96080803
	s_lshl_b32 s9, s2, 12                                      // 000000002594: 84098C02
	s_add_u32 s8, s9, s8                                       // 000000002598: 80080809
	s_add_u32 s8, s20, s8                                      // 00000000259C: 80080814
	s_mov_b32 s9, 0                                            // 0000000025A0: BE890080
	s_addc_u32 s9, s21, s9                                     // 0000000025A4: 82090915
	s_waitcnt lgkmcnt(0)                                       // 0000000025A8: BF89FC07
	global_store_b64 v1, v[2:3], s[8:9]                        // 0000000025AC: DC6E0000 00080201
	s_mov_b64 s[34:35], exec                                   // 0000000025B4: BEA2017E
	s_and_b64 s[34:35], s[28:29], s[34:35]                     // 0000000025B8: 8BA2221C
	s_cbranch_scc0 12                                          // 0000000025BC: BFA1000C <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x15f0>
	s_mov_b32 s34, 0x24000                                     // 0000000025C0: BEA200FF 00024000
	s_mul_i32 s34, s3, s34                                     // 0000000025C8: 96222203
	s_lshl_b32 s35, s2, 12                                     // 0000000025CC: 84238C02
	s_add_u32 s34, s35, s34                                    // 0000000025D0: 80222223
	s_waitcnt_vscnt null, 0x0                                  // 0000000025D4: BC7C0000
	s_add_u32 s8, s20, s34                                     // 0000000025D8: 80082214
	s_mov_b32 s9, 0                                            // 0000000025DC: BE890080
	s_addc_u32 s9, s21, s9                                     // 0000000025E0: 82090915
	s_waitcnt lgkmcnt(0)                                       // 0000000025E4: BF89FC07
	global_store_b64 v1, v[4:5], s[8:9] offset:256             // 0000000025E8: DC6E0100 00080401
	s_mov_b64 s[34:35], exec                                   // 0000000025F0: BEA2017E
	s_and_b64 s[34:35], s[30:31], s[34:35]                     // 0000000025F4: 8BA2221E
	s_cbranch_scc0 12                                          // 0000000025F8: BFA1000C <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x162c>
	s_mov_b32 s34, 0x24000                                     // 0000000025FC: BEA200FF 00024000
	s_mul_i32 s34, s3, s34                                     // 000000002604: 96222203
	s_lshl_b32 s35, s2, 12                                     // 000000002608: 84238C02
	s_add_u32 s34, s35, s34                                    // 00000000260C: 80222223
	s_waitcnt_vscnt null, 0x0                                  // 000000002610: BC7C0000
	s_add_u32 s8, s20, s34                                     // 000000002614: 80082214
	s_mov_b32 s9, 0                                            // 000000002618: BE890080
	s_addc_u32 s9, s21, s9                                     // 00000000261C: 82090915
	s_waitcnt lgkmcnt(0)                                       // 000000002620: BF89FC07
	global_store_b64 v1, v[18:19], s[8:9] offset:512           // 000000002624: DC6E0200 00081201
	s_mov_b64 s[34:35], exec                                   // 00000000262C: BEA2017E
	s_and_b64 s[34:35], s[32:33], s[34:35]                     // 000000002630: 8BA22220
	s_cbranch_scc0 865                                         // 000000002634: BFA10361 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x23bc>
	s_mov_b32 s34, 0x24000                                     // 000000002638: BEA200FF 00024000
	s_mul_i32 s34, s3, s34                                     // 000000002640: 96222203
	s_lshl_b32 s35, s2, 12                                     // 000000002644: 84238C02
	s_add_u32 s34, s35, s34                                    // 000000002648: 80222223
	s_waitcnt_vscnt null, 0x0                                  // 00000000264C: BC7C0000
	s_add_u32 s8, s20, s34                                     // 000000002650: 80082214
	s_mov_b32 s9, 0                                            // 000000002654: BE890080
	s_addc_u32 s9, s21, s9                                     // 000000002658: 82090915
	s_waitcnt lgkmcnt(0)                                       // 00000000265C: BF89FC07
	global_store_b64 v1, v[20:21], s[8:9] offset:768           // 000000002660: DC6E0300 00081401
	s_branch 852                                               // 000000002668: BFA00354 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x23bc>
	s_mov_b64 s[34:35], exec                                   // 00000000266C: BEA2017E
	s_and_b64 s[24:25], s[24:25], s[34:35]                     // 000000002670: 8B982218
	s_cbranch_scc0 20                                          // 000000002674: BFA10014 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x16c8>
	v_lshlrev_b32_e32 v1, 4, v10                               // 000000002678: 30021484
	s_mov_b32 s24, 0x900                                       // 00000000267C: BE9800FF 00000900
	s_mul_i32 s24, s3, s24                                     // 000000002684: 96181803
	s_lshl_b32 s25, s2, 6                                      // 000000002688: 84198602
	s_add_u32 s25, s25, s24                                    // 00000000268C: 80191819
	s_waitcnt_vscnt null, 0x0                                  // 000000002690: BC7C0000
	s_add_u32 s8, s16, s25                                     // 000000002694: 80081910
	s_mov_b32 s25, 0                                           // 000000002698: BE990080
	s_addc_u32 s9, s17, s25                                    // 00000000269C: 82091911
	s_waitcnt lgkmcnt(0)                                       // 0000000026A0: BF89FC07
	global_store_b32 v1, v7, s[8:9]                            // 0000000026A4: DC6A0000 00080701
	s_lshl_b32 s34, s2, 6                                      // 0000000026AC: 84228602
	s_add_u32 s24, s34, s24                                    // 0000000026B0: 80181822
	s_waitcnt_vscnt null, 0x0                                  // 0000000026B4: BC7C0000
	s_add_u32 s8, s18, s24                                     // 0000000026B8: 80081812
	s_addc_u32 s9, s19, s25                                    // 0000000026BC: 82091913
	global_store_b32 v1, v13, s[8:9]                           // 0000000026C0: DC6A0000 00080D01
	s_mov_b64 s[24:25], exec                                   // 0000000026C8: BE98017E
	s_and_b64 s[24:25], s[28:29], s[24:25]                     // 0000000026CC: 8B98181C
	s_cbranch_scc0 20                                          // 0000000026D0: BFA10014 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1724>
	v_lshlrev_b32_e32 v1, 4, v10                               // 0000000026D4: 30021484
	s_mov_b32 s24, 0x900                                       // 0000000026D8: BE9800FF 00000900
	s_mul_i32 s24, s3, s24                                     // 0000000026E0: 96181803
	s_lshl_b32 s25, s2, 6                                      // 0000000026E4: 84198602
	s_add_u32 s25, s25, s24                                    // 0000000026E8: 80191819
	s_waitcnt_vscnt null, 0x0                                  // 0000000026EC: BC7C0000
	s_add_u32 s8, s16, s25                                     // 0000000026F0: 80081910
	s_mov_b32 s25, 0                                           // 0000000026F4: BE990080
	s_addc_u32 s9, s17, s25                                    // 0000000026F8: 82091911
	s_waitcnt lgkmcnt(0)                                       // 0000000026FC: BF89FC07
	global_store_b32 v1, v8, s[8:9] offset:4                   // 000000002700: DC6A0004 00080801
	s_lshl_b32 s28, s2, 6                                      // 000000002708: 841C8602
	s_add_u32 s24, s28, s24                                    // 00000000270C: 8018181C
	s_waitcnt_vscnt null, 0x0                                  // 000000002710: BC7C0000
	s_add_u32 s8, s18, s24                                     // 000000002714: 80081812
	s_addc_u32 s9, s19, s25                                    // 000000002718: 82091913
	global_store_b32 v1, v14, s[8:9] offset:4                  // 00000000271C: DC6A0004 00080E01
	s_mov_b64 s[24:25], exec                                   // 000000002724: BE98017E
	s_and_b64 s[24:25], s[30:31], s[24:25]                     // 000000002728: 8B98181E
	s_cbranch_scc0 20                                          // 00000000272C: BFA10014 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1780>
	v_lshlrev_b32_e32 v1, 4, v10                               // 000000002730: 30021484
	s_mov_b32 s24, 0x900                                       // 000000002734: BE9800FF 00000900
	s_mul_i32 s24, s3, s24                                     // 00000000273C: 96181803
	s_lshl_b32 s25, s2, 6                                      // 000000002740: 84198602
	s_add_u32 s25, s25, s24                                    // 000000002744: 80191819
	s_waitcnt_vscnt null, 0x0                                  // 000000002748: BC7C0000
	s_add_u32 s8, s16, s25                                     // 00000000274C: 80081910
	s_mov_b32 s25, 0                                           // 000000002750: BE990080
	s_addc_u32 s9, s17, s25                                    // 000000002754: 82091911
	s_waitcnt lgkmcnt(0)                                       // 000000002758: BF89FC07
	global_store_b32 v1, v9, s[8:9] offset:8                   // 00000000275C: DC6A0008 00080901
	s_lshl_b32 s28, s2, 6                                      // 000000002764: 841C8602
	s_add_u32 s24, s28, s24                                    // 000000002768: 8018181C
	s_waitcnt_vscnt null, 0x0                                  // 00000000276C: BC7C0000
	s_add_u32 s8, s18, s24                                     // 000000002770: 80081812
	s_addc_u32 s9, s19, s25                                    // 000000002774: 82091913
	global_store_b32 v1, v15, s[8:9] offset:8                  // 000000002778: DC6A0008 00080F01
	s_mov_b64 s[24:25], exec                                   // 000000002780: BE98017E
	s_and_b64 s[24:25], s[32:33], s[24:25]                     // 000000002784: 8B981820
	s_cbranch_scc0 787                                         // 000000002788: BFA10313 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x23d8>
	v_lshlrev_b32_e32 v1, 4, v10                               // 00000000278C: 30021484
	s_mov_b32 s24, 0x900                                       // 000000002790: BE9800FF 00000900
	s_mul_i32 s24, s3, s24                                     // 000000002798: 96181803
	s_lshl_b32 s25, s2, 6                                      // 00000000279C: 84198602
	s_add_u32 s25, s25, s24                                    // 0000000027A0: 80191819
	s_waitcnt_vscnt null, 0x0                                  // 0000000027A4: BC7C0000
	s_add_u32 s8, s16, s25                                     // 0000000027A8: 80081910
	s_mov_b32 s25, 0                                           // 0000000027AC: BE990080
	s_addc_u32 s9, s17, s25                                    // 0000000027B0: 82091911
	s_waitcnt lgkmcnt(0)                                       // 0000000027B4: BF89FC07
	global_store_b32 v1, v12, s[8:9] offset:12                 // 0000000027B8: DC6A000C 00080C01
	s_lshl_b32 s2, s2, 6                                       // 0000000027C0: 84028602
	s_add_u32 s2, s2, s24                                      // 0000000027C4: 80021802
	s_waitcnt_vscnt null, 0x0                                  // 0000000027C8: BC7C0000
	s_add_u32 s8, s18, s2                                      // 0000000027CC: 80080212
	s_addc_u32 s9, s19, s25                                    // 0000000027D0: 82091913
	global_store_b32 v1, v17, s[8:9] offset:12                 // 0000000027D4: DC6A000C 00081101
	s_branch 766                                               // 0000000027DC: BFA002FE <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x23d8>
	v_lshlrev_b32_e32 v6, lit(0x2), v2                         // 0000000027E0: 300C04FF 00000002
	v_lshlrev_b32_e32 v7, lit(0x2), v1                         // 0000000027E8: 300E02FF 00000002
	v_mov_b32_e32 v8, 0x80                                     // 0000000027F0: 7E1002FF 00000080
	v_cmp_lt_u32_e64 s0, v1, 32                                // 0000000027F8: D4490000 02014101
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002800: BF870004
	v_add_nc_u32_e32 v9, lit(0x1), v6                          // 000000002804: 4A120CFF 00000001
	v_add_nc_u32_e32 v10, lit(0x2), v6                         // 00000000280C: 4A140CFF 00000002
	v_add_nc_u32_e32 v11, lit(0x3), v6                         // 000000002814: 4A160CFF 00000003
	v_lshl_add_u32 v6, s3, 3, v6                               // 00000000281C: D6460006 04190603
	v_cmp_lt_u32_e64 s24, v7, v8                               // 000000002824: D4490018 02021107
	v_lshl_add_u32 v7, s3, 3, v9                               // 00000000282C: D6460007 04250603
	v_lshl_add_u32 v8, s3, 3, v10                              // 000000002834: D6460008 04290603
	v_lshl_add_u32 v9, s3, 3, v11                              // 00000000283C: D6460009 042D0603
	v_cmp_lt_u32_e64 s28, v6, 32                               // 000000002844: D449001C 02014106
	s_and_b64 s[0:1], s[0:1], s[24:25]                         // 00000000284C: 8B801800
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000002850: BF870214
	v_cmp_lt_u32_e64 s24, v7, 32                               // 000000002854: D4490018 02014107
	v_cmp_lt_u32_e64 s30, v8, 32                               // 00000000285C: D449001E 02014108
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002864: BF870004
	v_cmp_lt_u32_e64 s32, v9, 32                               // 000000002868: D4490020 02014109
	s_and_saveexec_b64 s[0:1], s[0:1]                          // 000000002870: BE802100
	s_cbranch_scc0 730                                         // 000000002874: BFA102DA <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x23e0>
	s_mov_b64 s[34:35], exec                                   // 000000002878: BEA2017E
	s_and_b64 s[34:35], s[28:29], s[34:35]                     // 00000000287C: 8BA2221C
	s_cbranch_scc0 14                                          // 000000002880: BFA1000E <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x18bc>
	v_lshl_add_u32 v6, v2, 7, v1                               // 000000002884: D6460006 04050F02
	s_mov_b32 s34, 0x24000                                     // 00000000288C: BEA200FF 00024000
	s_mul_i32 s34, s3, s34                                     // 000000002894: 96222203
	s_lshl_b32 s35, s2, 12                                     // 000000002898: 84238C02
	s_add_u32 s34, s35, s34                                    // 00000000289C: 80222223
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000028A0: BF870001
	v_lshlrev_b32_e32 v6, 3, v6                                // 0000000028A4: 300C0C83
	s_add_u32 s8, s20, s34                                     // 0000000028A8: 80082214
	s_mov_b32 s9, 0                                            // 0000000028AC: BE890080
	s_addc_u32 s9, s21, s9                                     // 0000000028B0: 82090915
	global_store_b64 v6, v[4:5], s[8:9]                        // 0000000028B4: DC6E0000 00080406
	s_mov_b64 s[34:35], exec                                   // 0000000028BC: BEA2017E
	s_and_b64 s[34:35], s[24:25], s[34:35]                     // 0000000028C0: 8BA22218
	s_cbranch_scc0 15                                          // 0000000028C4: BFA1000F <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1904>
	v_lshl_add_u32 v6, v2, 7, v1                               // 0000000028C8: D6460006 04050F02
	s_mov_b32 s34, 0x24000                                     // 0000000028D0: BEA200FF 00024000
	s_mul_i32 s34, s3, s34                                     // 0000000028D8: 96222203
	s_lshl_b32 s35, s2, 12                                     // 0000000028DC: 84238C02
	s_add_u32 s34, s35, s34                                    // 0000000028E0: 80222223
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000028E4: BF870001
	v_lshlrev_b32_e32 v6, 3, v6                                // 0000000028E8: 300C0C83
	s_waitcnt_vscnt null, 0x0                                  // 0000000028EC: BC7C0000
	s_add_u32 s8, s20, s34                                     // 0000000028F0: 80082214
	s_mov_b32 s9, 0                                            // 0000000028F4: BE890080
	s_addc_u32 s9, s21, s9                                     // 0000000028F8: 82090915
	global_store_b64 v6, v[4:5], s[8:9] offset:256             // 0000000028FC: DC6E0100 00080406
	s_mov_b64 s[34:35], exec                                   // 000000002904: BEA2017E
	s_and_b64 s[34:35], s[30:31], s[34:35]                     // 000000002908: 8BA2221E
	s_cbranch_scc0 15                                          // 00000000290C: BFA1000F <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x194c>
	v_lshl_add_u32 v6, v2, 7, v1                               // 000000002910: D6460006 04050F02
	s_mov_b32 s34, 0x24000                                     // 000000002918: BEA200FF 00024000
	s_mul_i32 s34, s3, s34                                     // 000000002920: 96222203
	s_lshl_b32 s35, s2, 12                                     // 000000002924: 84238C02
	s_add_u32 s34, s35, s34                                    // 000000002928: 80222223
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000292C: BF870001
	v_lshlrev_b32_e32 v6, 3, v6                                // 000000002930: 300C0C83
	s_waitcnt_vscnt null, 0x0                                  // 000000002934: BC7C0000
	s_add_u32 s8, s20, s34                                     // 000000002938: 80082214
	s_mov_b32 s9, 0                                            // 00000000293C: BE890080
	s_addc_u32 s9, s21, s9                                     // 000000002940: 82090915
	global_store_b64 v6, v[4:5], s[8:9] offset:512             // 000000002944: DC6E0200 00080406
	s_mov_b64 s[34:35], exec                                   // 00000000294C: BEA2017E
	s_and_b64 s[34:35], s[32:33], s[34:35]                     // 000000002950: 8BA22220
	s_cbranch_scc0 674                                         // 000000002954: BFA102A2 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x23e0>
	v_lshl_add_u32 v6, v2, 7, v1                               // 000000002958: D6460006 04050F02
	s_mov_b32 s34, 0x24000                                     // 000000002960: BEA200FF 00024000
	s_mul_i32 s34, s3, s34                                     // 000000002968: 96222203
	s_lshl_b32 s35, s2, 12                                     // 00000000296C: 84238C02
	s_add_u32 s34, s35, s34                                    // 000000002970: 80222223
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002974: BF870001
	v_lshlrev_b32_e32 v6, 3, v6                                // 000000002978: 300C0C83
	s_waitcnt_vscnt null, 0x0                                  // 00000000297C: BC7C0000
	s_add_u32 s8, s20, s34                                     // 000000002980: 80082214
	s_mov_b32 s9, 0                                            // 000000002984: BE890080
	s_addc_u32 s9, s21, s9                                     // 000000002988: 82090915
	global_store_b64 v6, v[4:5], s[8:9] offset:768             // 00000000298C: DC6E0300 00080406
	s_branch 658                                               // 000000002994: BFA00292 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x23e0>
	s_mov_b64 s[34:35], exec                                   // 000000002998: BEA2017E
	s_and_b64 s[28:29], s[28:29], s[34:35]                     // 00000000299C: 8B9C221C
	s_cbranch_scc0 20                                          // 0000000029A0: BFA10014 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x19f4>
	v_lshlrev_b32_e32 v1, 4, v2                                // 0000000029A4: 30020484
	s_mov_b32 s28, 0x900                                       // 0000000029A8: BE9C00FF 00000900
	s_mul_i32 s28, s3, s28                                     // 0000000029B0: 961C1C03
	s_lshl_b32 s29, s2, 6                                      // 0000000029B4: 841D8602
	s_add_u32 s29, s29, s28                                    // 0000000029B8: 801D1C1D
	s_waitcnt_vscnt null, 0x0                                  // 0000000029BC: BC7C0000
	s_add_u32 s8, s16, s29                                     // 0000000029C0: 80081D10
	s_mov_b32 s29, 0                                           // 0000000029C4: BE9D0080
	s_addc_u32 s9, s17, s29                                    // 0000000029C8: 82091D11
	s_waitcnt lgkmcnt(0)                                       // 0000000029CC: BF89FC07
	global_store_b32 v1, v3, s[8:9]                            // 0000000029D0: DC6A0000 00080301
	s_lshl_b32 s34, s2, 6                                      // 0000000029D8: 84228602
	s_add_u32 s28, s34, s28                                    // 0000000029DC: 801C1C22
	s_waitcnt_vscnt null, 0x0                                  // 0000000029E0: BC7C0000
	s_add_u32 s8, s18, s28                                     // 0000000029E4: 80081C12
	s_addc_u32 s9, s19, s29                                    // 0000000029E8: 82091D13
	global_store_b32 v1, v4, s[8:9]                            // 0000000029EC: DC6A0000 00080401
	s_mov_b64 s[28:29], exec                                   // 0000000029F4: BE9C017E
	s_and_b64 s[24:25], s[24:25], s[28:29]                     // 0000000029F8: 8B981C18
	s_cbranch_scc0 20                                          // 0000000029FC: BFA10014 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1a50>
	v_lshlrev_b32_e32 v1, 4, v2                                // 000000002A00: 30020484
	s_mov_b32 s24, 0x900                                       // 000000002A04: BE9800FF 00000900
	s_mul_i32 s24, s3, s24                                     // 000000002A0C: 96181803
	s_lshl_b32 s25, s2, 6                                      // 000000002A10: 84198602
	s_add_u32 s25, s25, s24                                    // 000000002A14: 80191819
	s_waitcnt_vscnt null, 0x0                                  // 000000002A18: BC7C0000
	s_add_u32 s8, s16, s25                                     // 000000002A1C: 80081910
	s_mov_b32 s25, 0                                           // 000000002A20: BE990080
	s_addc_u32 s9, s17, s25                                    // 000000002A24: 82091911
	s_waitcnt lgkmcnt(0)                                       // 000000002A28: BF89FC07
	global_store_b32 v1, v3, s[8:9] offset:4                   // 000000002A2C: DC6A0004 00080301
	s_lshl_b32 s28, s2, 6                                      // 000000002A34: 841C8602
	s_add_u32 s24, s28, s24                                    // 000000002A38: 8018181C
	s_waitcnt_vscnt null, 0x0                                  // 000000002A3C: BC7C0000
	s_add_u32 s8, s18, s24                                     // 000000002A40: 80081812
	s_addc_u32 s9, s19, s25                                    // 000000002A44: 82091913
	global_store_b32 v1, v4, s[8:9] offset:4                   // 000000002A48: DC6A0004 00080401
	s_mov_b64 s[24:25], exec                                   // 000000002A50: BE98017E
	s_and_b64 s[24:25], s[30:31], s[24:25]                     // 000000002A54: 8B98181E
	s_cbranch_scc0 20                                          // 000000002A58: BFA10014 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1aac>
	v_lshlrev_b32_e32 v1, 4, v2                                // 000000002A5C: 30020484
	s_mov_b32 s24, 0x900                                       // 000000002A60: BE9800FF 00000900
	s_mul_i32 s24, s3, s24                                     // 000000002A68: 96181803
	s_lshl_b32 s25, s2, 6                                      // 000000002A6C: 84198602
	s_add_u32 s25, s25, s24                                    // 000000002A70: 80191819
	s_waitcnt_vscnt null, 0x0                                  // 000000002A74: BC7C0000
	s_add_u32 s8, s16, s25                                     // 000000002A78: 80081910
	s_mov_b32 s25, 0                                           // 000000002A7C: BE990080
	s_addc_u32 s9, s17, s25                                    // 000000002A80: 82091911
	s_waitcnt lgkmcnt(0)                                       // 000000002A84: BF89FC07
	global_store_b32 v1, v3, s[8:9] offset:8                   // 000000002A88: DC6A0008 00080301
	s_lshl_b32 s28, s2, 6                                      // 000000002A90: 841C8602
	s_add_u32 s24, s28, s24                                    // 000000002A94: 8018181C
	s_waitcnt_vscnt null, 0x0                                  // 000000002A98: BC7C0000
	s_add_u32 s8, s18, s24                                     // 000000002A9C: 80081812
	s_addc_u32 s9, s19, s25                                    // 000000002AA0: 82091913
	global_store_b32 v1, v4, s[8:9] offset:8                   // 000000002AA4: DC6A0008 00080401
	s_mov_b64 s[24:25], exec                                   // 000000002AAC: BE98017E
	s_and_b64 s[24:25], s[32:33], s[24:25]                     // 000000002AB0: 8B981820
	s_cbranch_scc0 592                                         // 000000002AB4: BFA10250 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x23f8>
	v_lshlrev_b32_e32 v1, 4, v2                                // 000000002AB8: 30020484
	s_mov_b32 s24, 0x900                                       // 000000002ABC: BE9800FF 00000900
	s_mul_i32 s24, s3, s24                                     // 000000002AC4: 96181803
	s_lshl_b32 s25, s2, 6                                      // 000000002AC8: 84198602
	s_add_u32 s25, s25, s24                                    // 000000002ACC: 80191819
	s_waitcnt_vscnt null, 0x0                                  // 000000002AD0: BC7C0000
	s_add_u32 s8, s16, s25                                     // 000000002AD4: 80081910
	s_mov_b32 s25, 0                                           // 000000002AD8: BE990080
	s_addc_u32 s9, s17, s25                                    // 000000002ADC: 82091911
	s_waitcnt lgkmcnt(0)                                       // 000000002AE0: BF89FC07
	global_store_b32 v1, v3, s[8:9] offset:12                  // 000000002AE4: DC6A000C 00080301
	s_lshl_b32 s2, s2, 6                                       // 000000002AEC: 84028602
	s_add_u32 s2, s2, s24                                      // 000000002AF0: 80021802
	s_waitcnt_vscnt null, 0x0                                  // 000000002AF4: BC7C0000
	s_add_u32 s8, s18, s2                                      // 000000002AF8: 80080212
	s_addc_u32 s9, s19, s25                                    // 000000002AFC: 82091913
	global_store_b32 v1, v4, s[8:9] offset:12                  // 000000002B00: DC6A000C 00080401
	s_branch 571                                               // 000000002B08: BFA0023B <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x23f8>
	s_waitcnt vmcnt(0)                                         // 000000002B0C: BF8903F7
	v_cmp_eq_i32_e64 s0, v0, 0                                 // 000000002B10: D4420000 02010100
	s_waitcnt_vscnt null, 0x0                                  // 000000002B18: BC7C0000
	s_barrier                                                  // 000000002B1C: BFBD0000
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002B20: BF870001
	s_and_saveexec_b64 s[24:25], s[0:1]                        // 000000002B24: BE982100
	s_cbranch_scc0 565                                         // 000000002B28: BFA10235 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2400>
	s_waitcnt vmcnt(0)                                         // 000000002B2C: BF8903F7
	v_mov_b32_e32 v1, 0                                        // 000000002B30: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 000000002B34: BF89FC07
	v_mov_b32_e32 v2, 1                                        // 000000002B38: 7E040281
	s_lshl_b32 s2, s3, 2                                       // 000000002B3C: 84028203
	s_mov_b32 s28, 0xffff                                      // 000000002B40: BE9C00FF 0000FFFF
	s_and_b32 s9, s23, s28                                     // 000000002B48: 8B091C17
	s_mov_b32 s10, 16                                          // 000000002B4C: BE8A0090
	s_mov_b32 s11, 0x31016000                                  // 000000002B50: BE8B00FF 31016000
	s_mov_b32 s8, s22                                          // 000000002B58: BE880016
	s_waitcnt_vscnt null, 0x0                                  // 000000002B5C: BC7C0000
	buffer_atomic_add_u32 v2, v1, s[8:11], s2 offen glc        // 000000002B60: E0D44000 02420201
	s_waitcnt vmcnt(0)                                         // 000000002B68: BF8903F7
	buffer_gl1_inv                                             // 000000002B6C: E0B00000 00000000
	buffer_gl0_inv                                             // 000000002B74: E0AC0000 00000000
	ds_store_b32 v1, v2 offset:17664                           // 000000002B7C: D8344500 00000201
	s_branch 542                                               // 000000002B84: BFA0021E <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2400>
	s_barrier                                                  // 000000002B88: BFBD0000
	s_waitcnt vmcnt(0)                                         // 000000002B8C: BF8903F7
	s_waitcnt lgkmcnt(0)                                       // 000000002B90: BF89FC07
	buffer_gl1_inv                                             // 000000002B94: E0B00000 00000000
	buffer_gl0_inv                                             // 000000002B9C: E0AC0000 00000000
	v_lshrrev_b32_e32 v1, 6, v0                                // 000000002BA4: 32020086
	v_and_b32_e32 v2, 63, v0                                   // 000000002BA8: 360400BF
	v_cmp_eq_i32_e64 s24, v0, 0                                // 000000002BAC: D4420018 02010100
	v_mov_b32_e32 v3, 0                                        // 000000002BB4: 7E060280
	v_mov_b32_e32 v4, 0xf149f2ca                               // 000000002BB8: 7E0802FF F149F2CA
	v_cmp_eq_i32_e64 s28, v1, 0                                // 000000002BC0: D442001C 02010101
	s_mov_b32 s30, 8                                           // 000000002BC8: BE9E0088
	s_mov_b32 s31, 0                                           // 000000002BCC: BE9F0080
	s_mov_b32 s32, s31                                         // 000000002BD0: BEA0001F
	s_cmp_lt_i32 s32, s30                                      // 000000002BD4: BF041E20
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000002BD8: BF870009
	s_cbranch_scc0 251                                         // 000000002BDC: BFA100FB <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1fcc>
	s_mov_b64 s[34:35], exec                                   // 000000002BE0: BEA2017E
	s_and_b64 s[36:37], s[28:29], s[34:35]                     // 000000002BE4: 8BA4221C
	s_cbranch_scc0 25                                          // 000000002BE8: BFA10019 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1c50>
	s_waitcnt lgkmcnt(0)                                       // 000000002BEC: BF89FC07
	v_mov_b32_e32 v1, v2                                       // 000000002BF0: 7E020302
	v_mov_b32_e32 v5, v4                                       // 000000002BF4: 7E0A0304
	v_cmp_lt_i32_e64 s36, v1, 36                               // 000000002BF8: D4410024 02014901
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002C00: BF870001
	s_and_saveexec_b64 s[36:37], s[36:37]                      // 000000002C04: BEA42124
	s_cbranch_scc0 523                                         // 000000002C08: BFA1020B <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2438>
	v_lshlrev_b32_e32 v6, 6, v1                                // 000000002C0C: 300C0286
	s_mov_b32 s33, 0x900                                       // 000000002C10: BEA100FF 00000900
	s_mul_i32 s33, s3, s33                                     // 000000002C18: 96212103
	s_lshl_b32 s36, s32, 2                                     // 000000002C1C: 84248220
	s_add_u32 s33, s33, s36                                    // 000000002C20: 80212421
	s_waitcnt_vscnt null, 0x0                                  // 000000002C24: BC7C0000
	s_add_u32 s8, s16, s33                                     // 000000002C28: 80082110
	s_mov_b32 s9, 0                                            // 000000002C2C: BE890080
	s_addc_u32 s9, s17, s9                                     // 000000002C30: 82090911
	global_load_b32 v6, v6, s[8:9]                             // 000000002C34: DC520000 06080006
	s_waitcnt lgkmcnt(0)                                       // 000000002C3C: BF89FC07
	v_add_nc_u32_e32 v1, 64, v1                                // 000000002C40: 4A0202C0
	s_waitcnt vmcnt(0)                                         // 000000002C44: BF8903F7
	v_max_f32_e32 v5, v5, v6                                   // 000000002C48: 200A0D05
	s_branch 65514                                             // 000000002C4C: BFA0FFEA <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1bf8>
	s_waitcnt lgkmcnt(0)                                       // 000000002C50: BF89FC07
	v_mov_b32_e32 v5, v4                                       // 000000002C54: 7E0A0304
	s_waitcnt lgkmcnt(0)                                       // 000000002C58: BF89FC07
	v_max_f32_dpp v1, v5, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000002C5C: 20020AFA FF08B105
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002C64: BF870091
	v_max_f32_dpp v1, v1, v1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000002C68: 200202FA FF084E01
	v_max_f32_dpp v1, v1, v1 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000002C70: 200202FA FF094101
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002C78: BF870091
	v_max_f32_dpp v1, v1, v1 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000002C7C: 200202FA FF094001
	v_permlanex16_b32 v5, v1, 0, 0                             // 000000002C84: D65C0005 02010101
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002C8C: BF870091
	v_max_f32_e32 v1, v1, v5                                   // 000000002C90: 20020B01
	v_readlane_b32 s10, v1, 0                                  // 000000002C94: D760000A 00010101
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002C9C: BF870091
	v_max_f32_e32 v1, s10, v1                                  // 000000002CA0: 2002020A
	v_readlane_b32 s10, v1, 63                                 // 000000002CA4: D760000A 00017F01
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002CAC: BF870001
	v_mov_b32_e32 v1, s10                                      // 000000002CB0: 7E02020A
	s_and_saveexec_b64 s[10:11], s[24:25]                      // 000000002CB4: BE8A2118
	s_cbranch_scc0 481                                         // 000000002CB8: BFA101E1 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2440>
	v_mov_b32_e32 v5, 0                                        // 000000002CBC: 7E0A0280
	ds_store_b32 v5, v1 offset:17672                           // 000000002CC0: D8344508 00000105
	s_branch 477                                               // 000000002CC8: BFA001DD <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2440>
	s_waitcnt lgkmcnt(0)                                       // 000000002CCC: BF89FC07
	v_mov_b32_e32 v5, v2                                       // 000000002CD0: 7E0A0302
	v_mov_b32_e32 v6, v3                                       // 000000002CD4: 7E0C0303
	v_cmp_lt_i32_e64 s34, v5, 36                               // 000000002CD8: D4410022 02014905
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002CE0: BF870001
	s_and_saveexec_b64 s[34:35], s[34:35]                      // 000000002CE4: BEA22122
	s_cbranch_scc0 479                                         // 000000002CE8: BFA101DF <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2468>
	v_lshlrev_b32_e32 v7, 6, v5                                // 000000002CEC: 300E0A86
	s_mov_b32 s33, 0x900                                       // 000000002CF0: BEA100FF 00000900
	s_mul_i32 s33, s3, s33                                     // 000000002CF8: 96212103
	s_lshl_b32 s34, s32, 2                                     // 000000002CFC: 84228220
	s_add_u32 s34, s33, s34                                    // 000000002D00: 80222221
	s_waitcnt_vscnt null, 0x0                                  // 000000002D04: BC7C0000
	s_add_u32 s8, s16, s34                                     // 000000002D08: 80082210
	s_mov_b32 s34, 0                                           // 000000002D0C: BEA20080
	s_addc_u32 s9, s17, s34                                    // 000000002D10: 82092211
	global_load_b32 v8, v7, s[8:9]                             // 000000002D14: DC520000 08080007
	s_lshl_b32 s35, s32, 2                                     // 000000002D1C: 84238220
	s_add_u32 s33, s33, s35                                    // 000000002D20: 80212321
	s_waitcnt vmcnt(0)                                         // 000000002D24: BF8903F7
	s_add_u32 s8, s18, s33                                     // 000000002D28: 80082112
	s_addc_u32 s9, s19, s34                                    // 000000002D2C: 82092213
	global_load_b32 v7, v7, s[8:9]                             // 000000002D30: DC520000 07080007
	s_waitcnt lgkmcnt(0)                                       // 000000002D38: BF89FC07
	v_add_nc_u32_e32 v5, 64, v5                                // 000000002D3C: 4A0A0AC0
	v_sub_f32_e32 v8, v8, v1                                   // 000000002D40: 08100308
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 000000002D44: BF8700D1
	v_mul_f32_e32 v8, 0x3fb8aa3b, v8                           // 000000002D48: 101010FF 3FB8AA3B
	v_exp_f32_e32 v8, v8                                       // 000000002D50: 7E104B08
	s_waitcnt_depctr depctr_hold_cnt(0) depctr_sa_sdst(0) depctr_va_vdst(0) depctr_va_sdst(0) depctr_va_ssrc(0) depctr_va_vcc(0) depctr_vm_vsrc(0)// 000000002D54: BF880000
	s_waitcnt vmcnt(0)                                         // 000000002D58: BF8903F7
	v_mul_f32_e32 v7, v7, v8                                   // 000000002D5C: 100E1107
	v_add_f32_e32 v6, v6, v7                                   // 000000002D60: 060C0F06
	s_branch 65500                                             // 000000002D64: BFA0FFDC <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1cd8>
	v_mov_b32_e32 v6, v3                                       // 000000002D68: 7E0C0303
	s_waitcnt lgkmcnt(0)                                       // 000000002D6C: BF89FC07
	v_add_f32_dpp v5, v6, v6 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000002D70: 060A0CFA FF08B106
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002D78: BF870091
	v_add_f32_dpp v5, v5, v5 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000002D7C: 060A0AFA FF084E05
	v_add_f32_dpp v5, v5, v5 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000002D84: 060A0AFA FF094105
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002D8C: BF870091
	v_add_f32_dpp v5, v5, v5 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000002D90: 060A0AFA FF094005
	v_permlanex16_b32 v6, v5, 0, 0                             // 000000002D98: D65C0006 02010105
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002DA0: BF870091
	v_add_f32_e32 v5, v5, v6                                   // 000000002DA4: 060A0D05
	v_readlane_b32 s10, v5, 0                                  // 000000002DA8: D760000A 00010105
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002DB0: BF870091
	v_add_f32_e32 v5, s10, v5                                  // 000000002DB4: 060A0A0A
	v_readlane_b32 s10, v5, 63                                 // 000000002DB8: D760000A 00017F05
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002DC0: BF870001
	v_mov_b32_e32 v5, s10                                      // 000000002DC4: 7E0A020A
	s_and_saveexec_b64 s[10:11], s[24:25]                      // 000000002DC8: BE8A2118
	s_cbranch_scc0 424                                         // 000000002DCC: BFA101A8 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2470>
	v_mov_b32_e32 v6, 0                                        // 000000002DD0: 7E0C0280
	ds_store_b32 v6, v5 offset:17676                           // 000000002DD4: D834450C 00000506
	s_branch 420                                               // 000000002DDC: BFA001A4 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2470>
	v_mov_b32_e32 v6, s31                                      // 000000002DE0: 7E0C021F
	s_mov_b64 s[34:35], exec                                   // 000000002DE4: BEA2017E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002DE8: BF870001
	v_mov_b32_e32 v7, v3                                       // 000000002DEC: 7E0E0303
	v_cmp_lt_i32_e64 s36, v6, 36                               // 000000002DF0: D4410024 02014906
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002DF8: BF870001
	s_and_saveexec_b64 s[36:37], s[36:37]                      // 000000002DFC: BEA42124
	s_cbranch_scc0 429                                         // 000000002E00: BFA101AD <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x24b8>
	v_lshlrev_b32_e32 v8, 6, v6                                // 000000002E04: 30100C86
	s_mov_b32 s33, 0x900                                       // 000000002E08: BEA100FF 00000900
	s_mul_i32 s33, s3, s33                                     // 000000002E10: 96212103
	s_lshl_b32 s36, s32, 2                                     // 000000002E14: 84248220
	s_add_u32 s36, s33, s36                                    // 000000002E18: 80242421
	s_mov_b32 s37, 0                                           // 000000002E1C: BEA50080
	s_waitcnt_vscnt null, 0x0                                  // 000000002E20: BC7C0000
	s_add_u32 s8, s16, s36                                     // 000000002E24: 80082410
	s_addc_u32 s9, s17, s37                                    // 000000002E28: 82092511
	global_load_b32 v9, v8, s[8:9]                             // 000000002E2C: DC520000 09080008
	s_lshl_b32 s36, s32, 2                                     // 000000002E34: 84248220
	s_add_u32 s36, s33, s36                                    // 000000002E38: 80242421
	s_waitcnt vmcnt(0)                                         // 000000002E3C: BF8903F7
	s_add_u32 s8, s16, s36                                     // 000000002E40: 80082410
	s_addc_u32 s9, s17, s37                                    // 000000002E44: 82092511
	global_load_b32 v10, v8, s[8:9] offset:64                  // 000000002E48: DC520040 0A080008
	s_lshl_b32 s36, s32, 2                                     // 000000002E50: 84248220
	s_add_u32 s36, s33, s36                                    // 000000002E54: 80242421
	v_lshl_add_u32 v11, v6, 11, v0                             // 000000002E58: D646000B 04011706
	s_waitcnt vmcnt(0)                                         // 000000002E60: BF8903F7
	s_add_u32 s8, s16, s36                                     // 000000002E64: 80082410
	s_addc_u32 s9, s17, s37                                    // 000000002E68: 82092511
	global_load_b32 v12, v8, s[8:9] offset:128                 // 000000002E6C: DC520080 0C080008
	s_lshl_b32 s36, s32, 2                                     // 000000002E74: 84248220
	s_add_u32 s33, s33, s36                                    // 000000002E78: 80212421
	s_waitcnt vmcnt(0)                                         // 000000002E7C: BF8903F7
	s_add_u32 s8, s16, s33                                     // 000000002E80: 80082110
	s_addc_u32 s9, s17, s37                                    // 000000002E84: 82092511
	global_load_b32 v8, v8, s[8:9] offset:192                  // 000000002E88: DC5200C0 08080008
	v_lshlrev_b32_e32 v11, 1, v11                              // 000000002E90: 30161681
	s_mov_b32 s33, 0x24000                                     // 000000002E94: BEA100FF 00024000
	s_mul_i32 s33, s3, s33                                     // 000000002E9C: 96212103
	s_lshl_b32 s36, s32, 8                                     // 000000002EA0: 84248820
	s_add_u32 s36, s33, s36                                    // 000000002EA4: 80242421
	s_waitcnt lgkmcnt(0)                                       // 000000002EA8: BF89FC07
	v_sub_f32_e32 v9, v9, v1                                   // 000000002EAC: 08120309
	s_waitcnt vmcnt(0)                                         // 000000002EB0: BF8903F7
	s_add_u32 s8, s20, s36                                     // 000000002EB4: 80082414
	s_addc_u32 s9, s21, s37                                    // 000000002EB8: 82092515
	global_load_d16_b16 v13, v11, s[8:9]                       // 000000002EBC: DC820000 0D08000B
	s_lshl_b32 s36, s32, 8                                     // 000000002EC4: 84248820
	s_add_u32 s36, s33, s36                                    // 000000002EC8: 80242421
	s_add_u32 s36, s36, 1                                      // 000000002ECC: 80248124
	v_sub_f32_e32 v10, v10, v1                                 // 000000002ED0: 0814030A
	v_mul_f32_e32 v9, 0x3fb8aa3b, v9                           // 000000002ED4: 101212FF 3FB8AA3B
	s_waitcnt vmcnt(0)                                         // 000000002EDC: BF8903F7
	s_add_u32 s8, s20, s36                                     // 000000002EE0: 80082414
	s_addc_u32 s9, s21, s37                                    // 000000002EE4: 82092515
	global_load_d16_b16 v14, v11, s[8:9] offset:4095           // 000000002EE8: DC820FFF 0E08000B
	s_lshl_b32 s36, s32, 8                                     // 000000002EF0: 84248820
	s_add_u32 s36, s33, s36                                    // 000000002EF4: 80242421
	s_mov_b32 s38, 0x1001                                      // 000000002EF8: BEA600FF 00001001
	s_add_u32 s36, s36, s38                                    // 000000002F00: 80242624
	v_sub_f32_e32 v12, v12, v1                                 // 000000002F04: 0818030C
	v_sub_f32_e32 v8, v8, v1                                   // 000000002F08: 08100308
	v_mul_f32_e32 v10, 0x3fb8aa3b, v10                         // 000000002F0C: 101414FF 3FB8AA3B
	v_exp_f32_e32 v9, v9                                       // 000000002F14: 7E124B09
	s_waitcnt vmcnt(0)                                         // 000000002F18: BF8903F7
	s_add_u32 s8, s20, s36                                     // 000000002F1C: 80082414
	s_addc_u32 s9, s21, s37                                    // 000000002F20: 82092515
	global_load_d16_b16 v15, v11, s[8:9] offset:4095           // 000000002F24: DC820FFF 0F08000B
	s_lshl_b32 s36, s32, 8                                     // 000000002F2C: 84248820
	s_add_u32 s33, s33, s36                                    // 000000002F30: 80212421
	s_mov_b32 s36, 0x2001                                      // 000000002F34: BEA400FF 00002001
	s_add_u32 s33, s33, s36                                    // 000000002F3C: 80212421
	v_fma_mix_f32 v9, v13, v9, lit(0x0) op_sel_hi:[1,0,0]      // 000000002F40: CC200009 0BFE130D 00000000
	s_waitcnt vmcnt(0)                                         // 000000002F4C: BF8903F7
	s_add_u32 s8, s20, s33                                     // 000000002F50: 80082114
	s_addc_u32 s9, s21, s37                                    // 000000002F54: 82092515
	global_load_d16_b16 v11, v11, s[8:9] offset:4095           // 000000002F58: DC820FFF 0B08000B
	v_mul_f32_e32 v12, 0x3fb8aa3b, v12                         // 000000002F60: 101818FF 3FB8AA3B
	v_add_f32_e32 v7, v7, v9                                   // 000000002F68: 060E1307
	v_exp_f32_e32 v9, v10                                      // 000000002F6C: 7E124B0A
	v_mul_f32_e32 v8, 0x3fb8aa3b, v8                           // 000000002F70: 101010FF 3FB8AA3B
	v_add_nc_u32_e32 v6, 4, v6                                 // 000000002F78: 4A0C0C84
	v_exp_f32_e32 v10, v12                                     // 000000002F7C: 7E144B0C
	v_exp_f32_e32 v8, v8                                       // 000000002F80: 7E104B08
	v_fma_mix_f32 v9, v14, v9, lit(0x0) op_sel_hi:[1,0,0]      // 000000002F84: CC200009 0BFE130E 00000000
	s_waitcnt_depctr depctr_hold_cnt(0) depctr_sa_sdst(0) depctr_va_vdst(0) depctr_va_sdst(0) depctr_va_ssrc(0) depctr_va_vcc(0) depctr_vm_vsrc(0)// 000000002F90: BF880000
	v_fma_mix_f32 v10, v15, v10, lit(0x0) op_sel_hi:[1,0,0]    // 000000002F94: CC20000A 0BFE150F 00000000
	s_delay_alu instid0(VALU_DEP_2)                            // 000000002FA0: BF870002
	v_add_f32_e32 v7, v7, v9                                   // 000000002FA4: 060E1307
	s_waitcnt vmcnt(0)                                         // 000000002FA8: BF8903F7
	v_fma_mix_f32 v8, v11, v8, lit(0x0) op_sel_hi:[1,0,0]      // 000000002FAC: CC200008 0BFE110B 00000000
	s_delay_alu instid0(VALU_DEP_2) | instid1(VALU_DEP_3)      // 000000002FB8: BF870182
	v_add_f32_e32 v7, v7, v10                                  // 000000002FBC: 060E1507
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_2)      // 000000002FC0: BF870101
	v_add_f32_e32 v7, v7, v8                                   // 000000002FC4: 060E1107
	s_branch 65417                                             // 000000002FC8: BFA0FF89 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1df0>
	s_waitcnt lgkmcnt(0)                                       // 000000002FCC: BF89FC07
	s_barrier                                                  // 000000002FD0: BFBD0000
	v_lshrrev_b32_e32 v1, lit(0x5), v0                         // 000000002FD4: 320200FF 00000005
	v_and_b32_e32 v2, 31, v0                                   // 000000002FDC: 3604009F
	s_lshl_b32 s10, s3, 12                                     // 000000002FE0: 840A8C03
	s_waitcnt_vscnt null, 0x0                                  // 000000002FE4: BC7C0000
	s_add_u32 s8, s4, s10                                      // 000000002FE8: 80080A04
	s_mov_b32 s9, 0                                            // 000000002FEC: BE890080
	s_addc_u32 s9, s5, s9                                      // 000000002FF0: 82090905
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_2)      // 000000002FF4: BF870101
	v_lshl_add_u32 v3, v1, 5, v2                               // 000000002FF8: D6460003 04090B01
	v_lshl_add_u32 v1, s3, 3, v1                               // 000000003000: D6460001 04050603
	v_lshlrev_b32_e32 v2, lit(0x2), v2                         // 000000003008: 300404FF 00000002
	v_and_b32_e32 v0, 7, v0                                    // 000000003010: 36000087
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003014: BF870004
	v_lshlrev_b32_e32 v3, 4, v3                                // 000000003018: 30060684
	s_delay_alu instid0(VALU_DEP_3) | instid1(VALU_DEP_4)      // 00000000301C: BF870203
	v_lshl_add_u32 v1, v1, 7, v2                               // 000000003020: D6460001 04090F01
	global_load_b128 v[4:7], v3, s[8:9]                        // 000000003028: DC5E0000 04080003
	s_waitcnt vmcnt(0)                                         // 000000003030: BF8903F7
	v_and_b32_e32 v2, 0x7fffffff, v4                           // 000000003034: 360408FF 7FFFFFFF
	v_and_b32_e32 v3, 0x7fffffff, v5                           // 00000000303C: 36060AFF 7FFFFFFF
	v_and_b32_e32 v8, 0x7fffffff, v6                           // 000000003044: 36100CFF 7FFFFFFF
	v_and_b32_e32 v9, 0x7fffffff, v7                           // 00000000304C: 36120EFF 7FFFFFFF
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003054: BF870004
	v_max_f32_e32 v2, lit(0x0), v2                             // 000000003058: 200404FF 00000000
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_4)      // 000000003060: BF870201
	v_max_f32_e32 v2, v2, v3                                   // 000000003064: 20040702
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_4)      // 000000003068: BF870201
	v_max_f32_e32 v2, v2, v8                                   // 00000000306C: 20041102
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_4)      // 000000003070: BF870201
	v_max_f32_e32 v2, v2, v9                                   // 000000003074: 20041302
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003078: BF870091
	v_mov_b32_dpp v3, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 00000000307C: 7E0602FA FF08B102
	v_max_f32_e32 v2, v2, v3                                   // 000000003084: 20040702
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003088: BF870091
	v_mov_b32_dpp v3, v2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 00000000308C: 7E0602FA FF084E02
	v_max_f32_e32 v2, v2, v3                                   // 000000003094: 20040702
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 000000003098: BF870241
	v_mov_b32_dpp v3, v2 row_xmask:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 00000000309C: 7E0602FA FF096402
	v_mov_b32_e32 v8, lit(0x3f800000)                          // 0000000030A4: 7E1002FF 3F800000
	v_lshrrev_b32_e32 v9, lit(0x5), v1                         // 0000000030AC: 321202FF 00000005
	v_lshrrev_b32_e32 v10, lit(0x7), v1                        // 0000000030B4: 321402FF 00000007
	v_max_f32_e32 v2, v2, v3                                   // 0000000030BC: 20040702
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000030C0: BF870113
	v_and_b32_e32 v3, 3, v9                                    // 0000000030C4: 36061283
	v_mul_f32_e32 v2, 0x3c010204, v2                           // 0000000030C8: 100404FF 3C010204
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000030D0: BF870122
	v_lshl_add_u32 v3, v3, 3, v0                               // 0000000030D4: D6460003 04010703
	v_rcp_f32_e32 v9, v2                                       // 0000000030DC: 7E125502
	v_cmp_lg_f32_e64 s4, v2, 0                                 // 0000000030E0: D4150004 02010102
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000030E8: BF870092
	v_mad_u32_u24 v3, v10, lit(0x24), v3                       // 0000000030EC: D60B0003 040DFF0A 00000024
	v_lshlrev_b32_e32 v3, 2, v3                                // 0000000030F8: 30060682
	s_waitcnt_depctr depctr_hold_cnt(0) depctr_sa_sdst(0) depctr_va_vdst(0) depctr_va_sdst(0) depctr_va_ssrc(0) depctr_va_vcc(0) depctr_vm_vsrc(0)// 0000000030FC: BF880000
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003100: BF870093
	v_cndmask_b32_e64 v9, 0, v9, s4                            // 000000003104: D5010009 00121280
	v_mul_f32_e32 v4, v4, v9                                   // 00000000310C: 10081304
	v_mul_f32_e32 v5, v5, v9                                   // 000000003110: 100A1305
	v_mul_f32_e32 v6, v6, v9                                   // 000000003114: 100C1306
	v_mul_f32_e32 v7, v7, v9                                   // 000000003118: 100E1307
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 00000000311C: BF870214
	v_trunc_f32_e32 v9, v4                                     // 000000003120: 7E124304
	v_trunc_f32_e32 v10, v5                                    // 000000003124: 7E144305
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000003128: BF870214
	v_trunc_f32_e32 v11, v6                                    // 00000000312C: 7E164306
	v_trunc_f32_e32 v12, v7                                    // 000000003130: 7E184307
	v_bfi_b32 v13, 0x7fffffff, v8, v4                          // 000000003134: D612000D 041210FF 7FFFFFFF
	v_sub_f32_e32 v4, v4, v9                                   // 000000003140: 08081304
	v_sub_f32_e32 v14, v5, v10                                 // 000000003144: 081C1505
	v_sub_f32_e32 v15, v6, v11                                 // 000000003148: 081E1706
	v_sub_f32_e32 v16, v7, v12                                 // 00000000314C: 08201907
	v_bfi_b32 v5, 0x7fffffff, v8, v5                           // 000000003150: D6120005 041610FF 7FFFFFFF
	v_and_b32_e32 v4, 0x7fffffff, v4                           // 00000000315C: 360808FF 7FFFFFFF
	v_and_b32_e32 v14, 0x7fffffff, v14                         // 000000003164: 361C1CFF 7FFFFFFF
	v_and_b32_e32 v15, 0x7fffffff, v15                         // 00000000316C: 361E1EFF 7FFFFFFF
	v_and_b32_e32 v16, 0x7fffffff, v16                         // 000000003174: 362020FF 7FFFFFFF
	v_bfi_b32 v6, 0x7fffffff, v8, v6                           // 00000000317C: D6120006 041A10FF 7FFFFFFF
	v_bfi_b32 v7, 0x7fffffff, v8, v7                           // 000000003188: D6120007 041E10FF 7FFFFFFF
	v_add_f32_e32 v8, v9, v13                                  // 000000003194: 06101B09
	v_add_f32_e32 v5, v10, v5                                  // 000000003198: 060A0B0A
	v_cmp_ge_f32_e64 s4, v4, 0.5                               // 00000000319C: D4160004 0201E104
	v_add_f32_e32 v4, v11, v6                                  // 0000000031A4: 06080D0B
	v_add_f32_e32 v6, v12, v7                                  // 0000000031A8: 060C0F0C
	v_cmp_ge_f32_e64 s10, v14, 0.5                             // 0000000031AC: D416000A 0201E10E
	v_cmp_ge_f32_e64 s24, v15, 0.5                             // 0000000031B4: D4160018 0201E10F
	v_cndmask_b32_e64 v7, v9, v8, s4                           // 0000000031BC: D5010007 00121109
	v_cmp_ge_f32_e64 s4, v16, 0.5                              // 0000000031C4: D4160004 0201E110
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000031CC: BF870214
	v_cndmask_b32_e64 v5, v10, v5, s10                         // 0000000031D0: D5010005 002A0B0A
	v_cndmask_b32_e64 v4, v11, v4, s24                         // 0000000031D8: D5010004 0062090B
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000031E0: BF870214
	v_cvt_i32_f32_e32 v8, v7                                   // 0000000031E4: 7E101107
	v_cndmask_b32_e64 v6, v12, v6, s4                          // 0000000031E8: D5010006 00120D0C
	v_add_f32_e32 v7, lit(0x0), v7                             // 0000000031F0: 060E0EFF 00000000
	v_cvt_i32_f32_e32 v9, v5                                   // 0000000031F8: 7E121105
	v_cvt_i32_f32_e32 v10, v4                                  // 0000000031FC: 7E141104
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000003200: BF870214
	v_cvt_i32_f32_e32 v11, v6                                  // 000000003204: 7E161106
	v_add_f32_e32 v5, v7, v5                                   // 000000003208: 060A0B07
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000320C: BF870004
	v_perm_b32 v7, v8, v9, lit(0x4)                            // 000000003210: D6440007 03FE1308 00000004
	s_delay_alu instid0(VALU_DEP_3) | instid1(VALU_DEP_4)      // 00000000321C: BF870203
	v_perm_b32 v8, v10, v11, lit(0x4)                          // 000000003220: D6440008 03FE170A 00000004
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000322C: BF870003
	v_add_f32_e32 v4, v5, v4                                   // 000000003230: 06080905
	s_delay_alu instid0(VALU_DEP_2) | instid1(VALU_DEP_3)      // 000000003234: BF870182
	v_perm_b32 v5, v7, v8, 0x1000504                           // 000000003238: D6440005 03FE1107 01000504
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000003244: BF8700B2
	v_add_f32_e32 v4, v4, v6                                   // 000000003248: 06080D04
	global_store_b32 v3, v5, s[6:7] offset:16                  // 00000000324C: DC6A0010 00060503
	v_mov_b32_dpp v3, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000003254: 7E0602FA FF08B104
	v_add_f32_e32 v3, v4, v3                                   // 00000000325C: 06060704
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003260: BF870091
	v_mov_b32_dpp v4, v3 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000003264: 7E0802FA FF084E03
	v_add_f32_e32 v3, v3, v4                                   // 00000000326C: 06060903
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000003270: BF870121
	v_mov_b32_dpp v4, v3 row_xmask:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000003274: 7E0802FA FF096403
	v_cmp_eq_i32_e64 s4, v0, 0                                 // 00000000327C: D4420004 02010100
	v_add_f32_e32 v0, v3, v4                                   // 000000003284: 06000903
	s_delay_alu instid0(VALU_DEP_2)                            // 000000003288: BF870002
	s_and_saveexec_b64 s[4:5], s[4:5]                          // 00000000328C: BE842104
	s_cbranch_scc0 153                                         // 000000003290: BFA10099 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x24f8>
	v_lshrrev_b32_e32 v3, lit(0x5), v1                         // 000000003294: 320602FF 00000005
	v_lshrrev_b32_e32 v1, lit(0x7), v1                         // 00000000329C: 320202FF 00000007
	v_cvt_f16_f32_e32 v4.l, v2                                 // 0000000032A4: 7E081502
	v_mul_f32_e32 v0, v0, v2                                   // 0000000032A8: 10000500
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000032AC: BF870194
	v_and_b32_e32 v2, 3, v3                                    // 0000000032B0: 36040683
	v_bfe_u32 v3, v4, 0, 16                                    // 0000000032B4: D6100003 02410104
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000032BC: BF870193
	v_cvt_f16_f32_e32 v0.l, v0                                 // 0000000032C0: 7E001500
	v_mad_u32_u24 v1, v1, lit(0x24), v2                        // 0000000032C4: D60B0001 0409FF01 00000024
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000032D0: BF870112
	v_bfe_u32 v0, v0, 0, 16                                    // 0000000032D4: D6100000 02410100
	v_lshlrev_b32_e32 v1, 2, v1                                // 0000000032DC: 30020282
	global_store_b16 v1, v3, s[6:7]                            // 0000000032E0: DC660000 00060301
	global_store_b16 v1, v0, s[6:7] offset:2                   // 0000000032E8: DC660002 00060001
	s_branch 129                                               // 0000000032F0: BFA00081 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x24f8>
	v_mov_b32_e32 v0, 0                                        // 0000000032F4: 7E000280
	s_lshl_b32 s3, s3, 2                                       // 0000000032F8: 84038203
	s_add_u32 s2, s22, s3                                      // 0000000032FC: 80020316
	s_mov_b32 s3, 0                                            // 000000003300: BE830080
	s_addc_u32 s3, s23, s3                                     // 000000003304: 82030317
	global_store_b32 v0, v0, s[2:3]                            // 000000003308: DC6A0000 00020000
	s_branch 127                                               // 000000003310: BFA0007F <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2510>
	s_waitcnt_vscnt null, 0x0                                  // 000000003314: BC7C0000
	s_endpgm                                                   // 000000003318: BFB00000
	s_and_b64 s[40:41], s[38:39], s[34:35]                     // 00000000331C: 8BA82226
	s_mov_b64 exec, s[38:39]                                   // 000000003320: BEFE0126
	s_xor_b64 exec, s[40:41], exec                             // 000000003324: 8DFE7E28
	s_cbranch_scc1 63574                                       // 000000003328: BFA2F856 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x484>
	s_branch 3                                                 // 00000000332C: BFA00003 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x233c>
	s_mov_b64 exec, s[38:39]                                   // 000000003330: BEFE0126
	v_mov_b32_e32 v1, 0                                        // 000000003334: 7E020280
	s_branch 65528                                             // 000000003338: BFA0FFF8 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x231c>
	s_mov_b64 exec, s[38:39]                                   // 00000000333C: BEFE0126
	s_branch 63571                                             // 000000003340: BFA0F853 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x490>
	s_and_b64 s[40:41], s[38:39], s[34:35]                     // 000000003344: 8BA82226
	s_mov_b64 exec, s[38:39]                                   // 000000003348: BEFE0126
	s_xor_b64 exec, s[40:41], exec                             // 00000000334C: 8DFE7E28
	s_cbranch_scc1 63607                                       // 000000003350: BFA2F877 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x530>
	s_branch 3                                                 // 000000003354: BFA00003 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2364>
	s_mov_b64 exec, s[38:39]                                   // 000000003358: BEFE0126
	v_mov_b32_e32 v1, 0                                        // 00000000335C: 7E020280
	s_branch 65528                                             // 000000003360: BFA0FFF8 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2344>
	s_mov_b64 exec, s[38:39]                                   // 000000003364: BEFE0126
	s_branch 63604                                             // 000000003368: BFA0F874 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x53c>
	s_and_b64 s[40:41], s[38:39], s[34:35]                     // 00000000336C: 8BA82226
	s_mov_b64 exec, s[38:39]                                   // 000000003370: BEFE0126
	s_xor_b64 exec, s[40:41], exec                             // 000000003374: 8DFE7E28
	s_cbranch_scc1 63640                                       // 000000003378: BFA2F898 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x5dc>
	s_branch 3                                                 // 00000000337C: BFA00003 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x238c>
	s_mov_b64 exec, s[38:39]                                   // 000000003380: BEFE0126
	v_mov_b32_e32 v1, 0                                        // 000000003384: 7E020280
	s_branch 65528                                             // 000000003388: BFA0FFF8 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x236c>
	s_mov_b64 exec, s[38:39]                                   // 00000000338C: BEFE0126
	s_branch 63637                                             // 000000003390: BFA0F895 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x5e8>
	s_and_b64 s[38:39], s[36:37], s[34:35]                     // 000000003394: 8BA62224
	s_mov_b64 exec, s[36:37]                                   // 000000003398: BEFE0124
	s_xor_b64 exec, s[38:39], exec                             // 00000000339C: 8DFE7E26
	s_cbranch_scc1 63672                                       // 0000000033A0: BFA2F8B8 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x684>
	s_branch 3                                                 // 0000000033A4: BFA00003 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x23b4>
	s_mov_b64 exec, s[36:37]                                   // 0000000033A8: BEFE0124
	v_mov_b32_e32 v1, 0                                        // 0000000033AC: 7E020280
	s_branch 65528                                             // 0000000033B0: BFA0FFF8 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2394>
	s_mov_b64 exec, s[36:37]                                   // 0000000033B4: BEFE0124
	s_branch 63669                                             // 0000000033B8: BFA0F8B5 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x690>
	s_mov_b64 exec, s[0:1]                                     // 0000000033BC: BEFE0100
	s_barrier                                                  // 0000000033C0: BFBD0000
	v_cmp_eq_i32_e64 s0, v11, 0                                // 0000000033C4: D4420000 0201010B
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000033CC: BF870001
	s_and_saveexec_b64 s[0:1], s[0:1]                          // 0000000033D0: BE802100
	s_cbranch_scc1 64677                                       // 0000000033D4: BFA2FCA5 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x166c>
	s_mov_b64 exec, s[0:1]                                     // 0000000033D8: BEFE0100
	s_branch 64971                                             // 0000000033DC: BFA0FDCB <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1b0c>
	s_mov_b64 exec, s[0:1]                                     // 0000000033E0: BEFE0100
	v_cmp_eq_i32_e64 s0, v1, 0                                 // 0000000033E4: D4420000 02010101
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000033EC: BF870001
	s_and_saveexec_b64 s[0:1], s[0:1]                          // 0000000033F0: BE802100
	s_cbranch_scc1 64872                                       // 0000000033F4: BFA2FD68 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1998>
	s_mov_b64 exec, s[0:1]                                     // 0000000033F8: BEFE0100
	s_branch 64963                                             // 0000000033FC: BFA0FDC3 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1b0c>
	s_mov_b64 exec, s[24:25]                                   // 000000003400: BEFE0118
	s_waitcnt lgkmcnt(0)                                       // 000000003404: BF89FC07
	s_barrier                                                  // 000000003408: BFBD0000
	v_mov_b32_e32 v1, 0                                        // 00000000340C: 7E020280
	s_mov_b64 s[10:11], exec                                   // 000000003410: BE8A017E
	ds_load_b32 v1, v1 offset:17664                            // 000000003414: D8D84500 01000001
	s_waitcnt lgkmcnt(0)                                       // 00000000341C: BF89FC07
	v_cmp_eq_i32_e64 s24, v1, 35                               // 000000003420: D4420018 02014701
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003428: BF870001
	s_and_b64 s[10:11], s[24:25], s[10:11]                     // 00000000342C: 8B8A0A18
	s_cbranch_scc1 64981                                       // 000000003430: BFA2FDD5 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1b88>
	s_branch 65463                                             // 000000003434: BFA0FFB7 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2314>
	s_mov_b64 exec, s[34:35]                                   // 000000003438: BEFE0122
	s_branch 65030                                             // 00000000343C: BFA0FE06 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1c58>
	s_mov_b64 exec, s[10:11]                                   // 000000003440: BEFE010A
	s_waitcnt lgkmcnt(0)                                       // 000000003444: BF89FC07
	s_barrier                                                  // 000000003448: BFBD0000
	v_mov_b32_e32 v5, 0                                        // 00000000344C: 7E0A0280
	s_mov_b64 s[10:11], exec                                   // 000000003450: BE8A017E
	s_and_b64 s[34:35], s[28:29], s[10:11]                     // 000000003454: 8BA20A1C
	ds_load_b32 v1, v5 offset:17672                            // 000000003458: D8D84508 01000005
	s_cbranch_scc1 65050                                       // 000000003460: BFA2FE1A <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1ccc>
	s_branch 65088                                             // 000000003464: BFA0FE40 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1d68>
	s_mov_b64 exec, s[10:11]                                   // 000000003468: BEFE010A
	s_branch 65087                                             // 00000000346C: BFA0FE3F <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1d6c>
	s_mov_b64 exec, s[10:11]                                   // 000000003470: BEFE010A
	s_waitcnt lgkmcnt(0)                                       // 000000003474: BF89FC07
	s_barrier                                                  // 000000003478: BFBD0000
	v_mov_b32_e32 v6, 0                                        // 00000000347C: 7E0C0280
	v_mov_b32_e32 v7, 0x80                                     // 000000003480: 7E0E02FF 00000080
	ds_load_b32 v5, v6 offset:17676                            // 000000003488: D8D8450C 05000006
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003490: BF870091
	v_cmp_lt_u32_e64 s10, v0, v7                               // 000000003494: D449000A 02020F00
	s_and_saveexec_b64 s[10:11], s[10:11]                      // 00000000349C: BE8A210A
	s_cbranch_scc1 65103                                       // 0000000034A0: BFA2FE4F <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1de0>
	s_mov_b64 exec, s[10:11]                                   // 0000000034A4: BEFE010A
	s_waitcnt lgkmcnt(0)                                       // 0000000034A8: BF89FC07
	s_barrier                                                  // 0000000034AC: BFBD0000
	s_add_u32 s32, s32, 1                                      // 0000000034B0: 80208120
	s_branch 64967                                             // 0000000034B4: BFA0FDC7 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x1bd4>
	s_mov_b64 exec, s[34:35]                                   // 0000000034B8: BEFE0122
	s_waitcnt lgkmcnt(0)                                       // 0000000034BC: BF89FC07
	v_rcp_f32_e32 v1, v5                                       // 0000000034C0: 7E025505
	v_lshlrev_b32_e32 v5, 2, v0                                // 0000000034C4: 300A0082
	s_lshl_b32 s33, s3, 12                                     // 0000000034C8: 84218C03
	s_lshl_b32 s34, s32, 9                                     // 0000000034CC: 84228920
	s_add_u32 s33, s33, s34                                    // 0000000034D0: 80212221
	s_waitcnt_vscnt null, 0x0                                  // 0000000034D4: BC7C0000
	s_add_u32 s8, s4, s33                                      // 0000000034D8: 80082104
	s_mov_b32 s9, 0                                            // 0000000034DC: BE890080
	s_addc_u32 s9, s5, s9                                      // 0000000034E0: 82090905
	s_waitcnt_depctr depctr_hold_cnt(0) depctr_sa_sdst(0) depctr_va_vdst(0) depctr_va_sdst(0) depctr_va_ssrc(0) depctr_va_vcc(0) depctr_vm_vsrc(0)// 0000000034E4: BF880000
	v_mul_f32_e32 v1, v7, v1                                   // 0000000034E8: 10020307
	global_store_b32 v5, v1, s[8:9]                            // 0000000034EC: DC6A0000 00080105
	s_branch 65515                                             // 0000000034F4: BFA0FFEB <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x24a4>
	s_waitcnt vmcnt(0)                                         // 0000000034F8: BF8903F7
	s_mov_b64 exec, s[4:5]                                     // 0000000034FC: BEFE0104
	s_waitcnt_vscnt null, 0x0                                  // 000000003500: BC7C0000
	s_barrier                                                  // 000000003504: BFBD0000
	s_and_saveexec_b64 s[0:1], s[0:1]                          // 000000003508: BE802100
	s_cbranch_scc1 65401                                       // 00000000350C: BFA2FF79 <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x22f4>
	s_mov_b64 exec, s[0:1]                                     // 000000003510: BEFE0100
	s_branch 65407                                             // 000000003514: BFA0FF7F <ggml_flash_attention_decode_split_f32_f16_wmma_next_q8+0x2314>
