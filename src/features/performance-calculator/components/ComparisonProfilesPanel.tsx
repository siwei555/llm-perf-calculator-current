import { getModelDefinition } from "../../../engines/model-registry";
import type { ComparisonProfile } from "../types/comparison";
import { getResourceScaling } from "../services/platformScaling";

type Props = {
  profiles: ComparisonProfile[];
  onAdd: () => void;
  onDuplicate: (profileId: string) => void;
  onDelete: (profileId: string) => void;
  onToggle: (profileId: string) => void;
  onUpdate: (profileId: string, field: "batchSize" | "chipCount" | "precision", value: number | "w4a8" | "w8a8" | "fp8" | "bf16" | "custom") => void;
  onCalculate: () => void;
  error?: string | null;
};

export function ComparisonProfilesPanel({
  profiles,
  onAdd,
  onDuplicate,
  onDelete,
  onToggle,
  onUpdate,
  onCalculate,
  error
}: Props) {
  return (
    <article className="panel comparison-profiles">
      <div className="comparison-profiles__header">
        <div>
          <h3>对比配置</h3>
          <p>配置保存后独立于当前输入；最多保留 4 条。</p>
        </div>
        <div className="comparison-profiles__actions">
          <button type="button" className="secondary-button" onClick={onAdd} disabled={profiles.length >= 4}>
            添加当前配置
          </button>
          <button type="button" className="primary-button" onClick={onCalculate} disabled={!profiles.some((profile) => profile.enabled)}>
            开始对比
          </button>
        </div>
      </div>
      <div className="comparison-profile-list">
        {profiles.map((profile) => {
          const model = getModelDefinition(profile.modelId);
          return (
            <div className={`comparison-profile${profile.enabled ? "" : " is-disabled"}`} key={profile.id}>
              <label className="comparison-profile__toggle" title={`NChip: ${profile.platform.chipCount}\nPrefill compute scaling: ${getResourceScaling(profile.platform, "prefill", "compute").toFixed(2)}x\nPrefill bandwidth scaling: ${getResourceScaling(profile.platform, "prefill", "bandwidth").toFixed(2)}x\nDecode compute scaling: ${getResourceScaling(profile.platform, "decode", "compute").toFixed(2)}x\nDecode bandwidth scaling: ${getResourceScaling(profile.platform, "decode", "bandwidth").toFixed(2)}x`}>
                <input type="checkbox" checked={profile.enabled} onChange={() => onToggle(profile.id)} />
                <i style={{ background: profile.color }} />
                <span>{profile.label}</span>
              </label>
              <small>
                {model.family} · {profile.platform.computeThroughputTflops} TFLOPS/chip · {profile.platform.memoryBandwidthGbps} GB/s/chip
              </small>
              <div className="comparison-profile__fields">
                <label>Batch<input type="number" min="1" value={profile.platform.batchSize} onChange={(event) => onUpdate(profile.id, "batchSize", Number(event.target.value))} /></label>
                <label>N Chip<input type="number" min="1" value={profile.platform.chipCount} onChange={(event) => onUpdate(profile.id, "chipCount", Number(event.target.value))} /></label>
                <label>Precision<select value={profile.precision} onChange={(event) => onUpdate(profile.id, "precision", event.target.value as "w4a8" | "w8a8" | "fp8" | "bf16" | "custom")}><option value="custom">Custom</option><option value="w4a8">W4A8</option><option value="w8a8">W8A8</option><option value="fp8">FP8</option><option value="bf16">BF16</option></select></label>
              </div>
              <div className="comparison-profile__buttons">
                <button type="button" onClick={() => onDuplicate(profile.id)} disabled={profiles.length >= 4}>复制</button>
                <button type="button" onClick={() => onDelete(profile.id)}>删除</button>
              </div>
            </div>
          );
        })}
      </div>
      {error ? <p className="field-error comparison-profiles__error">{error}</p> : null}
    </article>
  );
}
