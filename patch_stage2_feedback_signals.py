from __future__ import annotations

import py_compile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TARGET = PROJECT_ROOT / 'models' / 'EG_GM_PUNet_v2' / 'eg_gm_punet_v2_stage2_model.py'
BACKUP = TARGET.with_name(TARGET.name + '.bak_before_feedback_signal_fix')


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(
            f'没有找到 Stage 2 model 文件: {TARGET}\n'
            '请把本补丁脚本放到 E:/workspace/EG-GM-PUNet/ 项目根目录后再运行。'
        )

    text = TARGET.read_text(encoding='utf-8')

    old = '''        feedback_signals = {
            "ambiguity_demand": controls.ambiguity_demand,
            "categorical_capacity": controls.categorical_capacity,
            "categorical_error": controls.categorical_error,
        }
        for name in self.group_names:
            feedback_signals[f"{name}_capacity"] = controls.group_capacity[name]
            feedback_signals[f"{name}_error"] = controls.group_error[name]
'''

    new = '''        # Legacy logger compatibility: old train loop expects these four keys.
        # In Stage 2 they represent demand-capacity mismatch, which directly drives beta/tau.
        feedback_signals = {
            "categorical": controls.categorical_error.detach(),
            "coarse": controls.group_error["coarse"].detach(),
            "middle": controls.group_error["middle"].detach(),
            "fine": controls.group_error["fine"].detach(),
            "ambiguity_demand": controls.ambiguity_demand.detach(),
            "categorical_capacity": controls.categorical_capacity.detach(),
            "categorical_error": controls.categorical_error.detach(),
        }
        for name in self.group_names:
            feedback_signals[f"{name}_capacity"] = controls.group_capacity[name].detach()
            feedback_signals[f"{name}_error"] = controls.group_error[name].detach()
'''

    if '"categorical": controls.categorical_error.detach()' not in text:
        if old not in text:
            raise RuntimeError(
                '没有找到预期的 feedback_signals 代码块。请把当前 '
                'eg_gm_punet_v2_stage2_model.py 发给我核对。'
            )
        if not BACKUP.exists():
            BACKUP.write_text(text, encoding='utf-8')
            print(f'[BACKUP] {BACKUP}')
        text = text.replace(old, new, 1)
        TARGET.write_text(text, encoding='utf-8')
        print(f'[FIXED] {TARGET}')
    else:
        print('[INFO] 当前文件已经包含修复。')

    py_compile.compile(str(TARGET), doraise=True)
    print('[PASS] Python syntax check passed.')
    print('现在重新运行 Stage 2 smoke test。')


if __name__ == '__main__':
    main()
