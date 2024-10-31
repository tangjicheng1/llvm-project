//===-- AMDGPURegBankSelect.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Assign register banks to all register operands of G_ instructions using
/// machine uniformity analysis.
/// Sgpr - uniform values and some lane masks
/// Vgpr - divergent, non S1, values
/// Vcc  - divergent S1 values(lane masks)
/// However in some cases G_ instructions with this register bank assignment
/// can't be inst-selected. This is solved in AMDGPURegBankLegalize.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUGlobalISelUtils.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/CSEMIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineUniformityAnalysis.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "amdgpu-regbankselect"

using namespace llvm;
using namespace AMDGPU;

namespace {

class AMDGPURegBankSelect : public MachineFunctionPass {
public:
  static char ID;

  AMDGPURegBankSelect() : MachineFunctionPass(ID) {
    initializeAMDGPURegBankSelectPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Register Bank Select";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<GISelCSEAnalysisWrapperPass>();
    AU.addRequired<MachineUniformityAnalysisPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  // This pass assigns register banks to all virtual registers, and we maintain
  // this property in subsequent passes
  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::RegBankSelected);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPURegBankSelect, DEBUG_TYPE,
                      "AMDGPU Register Bank Select", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelCSEAnalysisWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineUniformityAnalysisPass)
INITIALIZE_PASS_END(AMDGPURegBankSelect, DEBUG_TYPE,
                    "AMDGPU Register Bank Select", false, false)

char AMDGPURegBankSelect::ID = 0;

char &llvm::AMDGPURegBankSelectID = AMDGPURegBankSelect::ID;

FunctionPass *llvm::createAMDGPURegBankSelectPass() {
  return new AMDGPURegBankSelect();
}

class RegBankSelectHelper {
  MachineIRBuilder &B;
  MachineRegisterInfo &MRI;
  AMDGPU::IntrinsicLaneMaskAnalyzer &ILMA;
  const MachineUniformityInfo &MUI;
  const SIRegisterInfo &TRI;
  const RegisterBank *SgprRB;
  const RegisterBank *VgprRB;
  const RegisterBank *VccRB;

public:
  RegBankSelectHelper(MachineIRBuilder &B,
                      AMDGPU::IntrinsicLaneMaskAnalyzer &ILMA,
                      const MachineUniformityInfo &MUI,
                      const SIRegisterInfo &TRI, const RegisterBankInfo &RBI)
      : B(B), MRI(*B.getMRI()), ILMA(ILMA), MUI(MUI), TRI(TRI),
        SgprRB(&RBI.getRegBank(AMDGPU::SGPRRegBankID)),
        VgprRB(&RBI.getRegBank(AMDGPU::VGPRRegBankID)),
        VccRB(&RBI.getRegBank(AMDGPU::VCCRegBankID)) {}

  bool shouldRegBankSelect(MachineInstr &MI) {
    return MI.isPreISelOpcode() || MI.isCopy();
  }

  // Temporal divergence copy: COPY to vgpr with implicit use of $exec inside of
  // the cycle
  // Note: uniformity analysis does not consider that registers with vgpr def
  // are divergent (you can have uniform value in vgpr).
  // - TODO: implicit use of $exec could be implemented as indicator that
  //   instruction is divergent
  bool isTemporalDivergenceCopy(Register Reg) {
    MachineInstr *MI = MRI.getVRegDef(Reg);
    if (!MI->isCopy())
      return false;

    for (auto Op : MI->implicit_operands()) {
      if (!Op.isReg())
        continue;

      if (Op.getReg() == TRI.getExec()) {
        return true;
      }
    }

    return false;
  }

  void setRBDef(MachineInstr &MI, MachineOperand &DefOP,
                const RegisterBank *RB) {
    Register Reg = DefOP.getReg();

    if (!MRI.getRegClassOrNull(Reg)) {
      MRI.setRegBank(Reg, *RB);
      return;
    }

    // Register that already has Register class got it during pre-inst selection
    // of another instruction. Maybe cross bank copy was required so we insert a
    // copy that can be removed later. This simplifies post regbanklegalize
    // combiner and avoids need to special case some patterns.
    LLT Ty = MRI.getType(Reg);
    Register NewReg = MRI.createVirtualRegister({RB, Ty});
    DefOP.setReg(NewReg);

    auto &MBB = *MI.getParent();
    B.setInsertPt(MBB, MBB.SkipPHIsAndLabels(std::next(MI.getIterator())));
    B.buildCopy(Reg, NewReg);

    // The problem was discovered for uniform S1 that was used as both
    // lane mask(vcc) and regular sgpr S1.
    // - lane-mask(vcc) use was by si_if, this use is divergent and requires
    //   non-trivial sgpr-S1-to-vcc copy. But pre-inst-selection of si_if sets
    //   sreg_64_xexec(S1) on def of uniform S1 making it lane-mask.
    // - the regular sgpr S1(uniform) instruction is now broken since
    //   it uses sreg_64_xexec(S1) which is divergent.

    // Replace virtual registers with register class on generic instructions
    // uses with virtual registers with register bank.
    SmallVector<MachineOperand *, 4> RegUsesOnGInstrs;
    for (auto &UseMI : MRI.use_instructions(Reg)) {
      if (shouldRegBankSelect(UseMI)) {
        for (MachineOperand &Op : UseMI.operands()) {
          if (Op.isReg() && Op.getReg() == Reg)
            RegUsesOnGInstrs.push_back(&Op);
        }
      }
    }
    for (MachineOperand *Op : RegUsesOnGInstrs) {
      Op->setReg(NewReg);
    }
  }

  Register tryGetVReg(MachineOperand &Op) {
    if (!Op.isReg())
      return {};

    Register Reg = Op.getReg();
    if (!Reg.isVirtual())
      return {};

    return Reg;
  }

  void assignBanksOnDefs(MachineInstr &MI) {
    if (!shouldRegBankSelect(MI))
      return;

    for (MachineOperand &DefOP : MI.defs()) {
      Register DefReg = tryGetVReg(DefOP);
      if (!DefReg.isValid())
        continue;

      // Copies can have register class on def registers.
      if (MI.isCopy() && MRI.getRegClassOrNull(DefReg)) {
        continue;
      }

      if (MUI.isUniform(DefReg) || ILMA.isS32S64LaneMask(DefReg)) {
        setRBDef(MI, DefOP, SgprRB);
      } else {
        if (MRI.getType(DefReg) == LLT::scalar(1))
          setRBDef(MI, DefOP, VccRB);
        else
          setRBDef(MI, DefOP, VgprRB);
      }
    }
  }

  void constrainRBUse(MachineInstr &MI, MachineOperand &UseOP,
                      const RegisterBank *RB) {
    Register Reg = UseOP.getReg();

    LLT Ty = MRI.getType(Reg);
    Register NewReg = MRI.createVirtualRegister({RB, Ty});
    UseOP.setReg(NewReg);

    if (MI.isPHI()) {
      auto DefMI = MRI.getVRegDef(Reg)->getIterator();
      MachineBasicBlock *DefMBB = DefMI->getParent();
      B.setInsertPt(*DefMBB, DefMBB->SkipPHIsAndLabels(std::next(DefMI)));
    } else {
      B.setInstr(MI);
    }

    B.buildCopy(NewReg, Reg);
  }

  void constrainBanksOnUses(MachineInstr &MI) {
    if (!shouldRegBankSelect(MI))
      return;

    // Copies can have register class on use registers.
    if (MI.isCopy())
      return;

    for (MachineOperand &UseOP : MI.uses()) {
      auto UseReg = tryGetVReg(UseOP);
      if (!UseReg.isValid())
        continue;

      // UseReg already has register bank.
      if (MRI.getRegBankOrNull(UseReg))
        continue;

      if (!isTemporalDivergenceCopy(UseReg) &&
          (MUI.isUniform(UseReg) || ILMA.isS32S64LaneMask(UseReg))) {
        constrainRBUse(MI, UseOP, SgprRB);
      } else {
        if (MRI.getType(UseReg) == LLT::scalar(1))
          constrainRBUse(MI, UseOP, VccRB);
        else
          constrainRBUse(MI, UseOP, VgprRB);
      }
    }
  }
};

bool AMDGPURegBankSelect::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;

  // Setup the instruction builder with CSE.
  const TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  GISelCSEAnalysisWrapper &Wrapper =
      getAnalysis<GISelCSEAnalysisWrapperPass>().getCSEWrapper();
  GISelCSEInfo &CSEInfo = Wrapper.get(TPC.getCSEConfig());
  GISelObserverWrapper Observer;
  Observer.addObserver(&CSEInfo);

  CSEMIRBuilder B(MF);
  B.setCSEInfo(&CSEInfo);
  B.setChangeObserver(Observer);

  RAIIDelegateInstaller DelegateInstaller(MF, &Observer);
  RAIIMFObserverInstaller MFObserverInstaller(MF, Observer);

  IntrinsicLaneMaskAnalyzer ILMA(MF);
  MachineUniformityInfo &MUI =
      getAnalysis<MachineUniformityAnalysisPass>().getUniformityInfo();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  RegBankSelectHelper RBSHelper(B, ILMA, MUI, *ST.getRegisterInfo(),
                                *ST.getRegBankInfo());

  // Assign register banks to ALL def registers on G_ instructions.
  // Same for copies if they have no register bank or class on def.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      RBSHelper.assignBanksOnDefs(MI);
    }
  }

  // At this point all virtual registers have register class or bank
  // - Defs of G_ instructions have register banks.
  // - Defs and uses of inst-selected instructions have register class.
  // - Defs and uses of copies can have either register class or bank
  //   and most notably:
  // - Uses of G_ instructions can have either register class or bank.

  // Reassign uses of G_ instructions to only have register banks.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      RBSHelper.constrainBanksOnUses(MI);
    }
  }

  // Defs and uses of G_ instructions have register banks exclusively.
  return true;
}
