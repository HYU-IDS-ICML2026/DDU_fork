# utils/sam.py
# ============================================================
# SAM(Sharpness-Aware Minimization) 옵티마이저 구현
# 참고 논문: Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021
# 참고 구현: https://github.com/davda54/sam (MIT License)
#  - 본 파일은 위 레포의 sam.py 구현 구조(first_step/second_step/step)를 DDU 코드베이스에 통합하기 위해
#    동일한 알고리즘/인터페이스를 따르도록 작성했습니다.
#  - MIT 라이선스 전문은 원본 레포를 따릅니다.
# ============================================================

import torch
import torch.nn as nn


class SAM(torch.optim.Optimizer):
    """
    SAM optimizer wrapper.
    - base_optimizer: torch.optim.SGD 같은 "기본 옵티마이저 클래스"를 넘깁니다.
    - rho: SAM neighborhood 크기
    - adaptive: ASAM 옵션(원본 구현의 adaptive 플래그)
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        # [출처] davda54/sam 방식: base_optimizer를 내부에 생성하고 param_groups를 공유
        # https://raw.githubusercontent.com/davda54/sam/main/sam.py  :contentReference[oaicite:2]{index=2}
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        """
        [SAM 1단계] 현재 w에서 loss를 증가시키는 방향으로 rho-이웃 내로 이동: w -> w + e(w)
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 원본 구현: old_p 저장 후 perturbation 적용
                self.state[p]["old_p"] = p.data.clone()
                e_w = ((torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad) * scale.to(p)
                p.add_(e_w)  # climb to local maximum "w + e(w)"
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True):
        """
        [SAM 2단계] 원래 파라미터(w)로 복귀 후, 그 지점의 그래디언트로 실제 업데이트 수행
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w"
        self.base_optimizer.step()  # sharpness-aware update
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure=None):
        """
        [래퍼] first_step -> (closure로 full forward/backward) -> second_step
        - 주의: closure는 "반드시" full forward/backward를 다시 계산해야 함.
        """
        assert closure is not None, "SAM requires closure, but it was not provided."
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)

    def _grad_norm(self):
        # [출처] davda54/sam: model parallel 가능성을 고려해 shared_device로 norm 집계 :contentReference[oaicite:3]{index=3}
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = (torch.abs(p) if group["adaptive"] else 1.0) * p.grad
                norms.append(g.norm(p=2).to(shared_device))
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# -------------------------
# BatchNorm running stats 토글 헬퍼
# -------------------------
def disable_running_stats(model: nn.Module):
    """
    [변경 사항] SAM 2nd pass에서 BN running stats가 또 업데이트되는 것을 방지
    - davda54/sam example train loop에서 널리 쓰이는 패턴
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.backup_momentum = m.momentum
            m.momentum = 0


def enable_running_stats(model: nn.Module):
    """
    disable_running_stats로 바꿔둔 BN momentum을 원복
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and hasattr(m, "backup_momentum"):
            m.momentum = m.backup_momentum
