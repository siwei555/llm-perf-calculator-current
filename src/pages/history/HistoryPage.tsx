import { useMemo, useState } from "react";
import type {
  CalculationHistoryRecord,
  HistoryTimeOrder
} from "../../domain/history/types";
import { useCalculatorContext } from "../../features/performance-calculator/state/CalculatorProvider";

function formatDate(value: string) {
  return new Date(value).toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  });
}

function formatNumber(value: number, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function HistoryRecordCard({
  record,
  onDelete
}: {
  record: CalculationHistoryRecord;
  onDelete: (record: CalculationHistoryRecord) => void;
}) {
  return (
    <details className="history-record">
      <summary className="history-record__summary">
        <div className="history-record__identity">
          <span className="history-record__date">{formatDate(record.createdAt)}</span>
          <strong>{record.modelDisplayName}</strong>
        </div>
        <div className="history-record__metrics">
          <span>
            <small>Prefill TPS</small>
            <strong>{formatNumber(record.result.prefillTps)}</strong>
          </span>
          <span>
            <small>Decode TPS</small>
            <strong>{formatNumber(record.result.decodeTps)}</strong>
          </span>
          <span>
            <small>Memory</small>
            <strong>{formatNumber(record.result.totalRuntimeMemoryGb)} GB</strong>
          </span>
        </div>
        <span className="history-record__chevron" aria-hidden="true" />
      </summary>

      <div className="history-record__details">
        <section>
          <h4>计算参数</h4>
          <dl className="history-detail-list">
            <div><dt>Prompt Tokens</dt><dd>{record.workload.prefillTokenLength.toLocaleString()}</dd></div>
            <div><dt>Decode Output Tokens</dt><dd>{record.workload.decodeOutputTokens?.toLocaleString() ?? "-"}</dd></div>
            <div><dt>Token Sweep</dt><dd>{record.workload.tokenRangeStart.toLocaleString()} – {record.workload.tokenRangeEnd.toLocaleString()} / {record.workload.tokenRangeStep.toLocaleString()}</dd></div>
            <div><dt>Compute</dt><dd>{record.platform.computeThroughputTflops} TFLOPS × {record.platform.computeEfficiency}</dd></div>
            <div><dt>Bandwidth</dt><dd>{record.platform.memoryBandwidthGbps} GB/s × {record.platform.bandwidthEfficiency}</dd></div>
            <div><dt>Memory Capacity</dt><dd>{record.platform.memoryCapacityGb} GB</dd></div>
            <div><dt>Batch Size</dt><dd>{record.platform.batchSize}</dd></div>
            <div><dt>Bytes / Weight</dt><dd>{record.platform.bytesPerWeight}</dd></div>
            <div><dt>Bytes / Activation</dt><dd>{record.platform.bytesPerActivation}</dd></div>
            <div><dt>Bytes / Expert</dt><dd>{record.platform.bytesPerExpert}</dd></div>
          </dl>
        </section>

        <section>
          <h4>计算结果</h4>
          <dl className="history-detail-list">
            <div><dt>TTFT</dt><dd>{formatNumber(record.result.ttftMs)} ms</dd></div>
            <div><dt>Prefill TPS</dt><dd>{formatNumber(record.result.prefillTps)} tokens/s</dd></div>
            <div><dt>Decode TPS</dt><dd>{formatNumber(record.result.decodeTps)} tokens/s</dd></div>
            <div><dt>Runtime Memory</dt><dd>{formatNumber(record.result.totalRuntimeMemoryGb)} GB</dd></div>
            <div><dt>Prefill Bottleneck</dt><dd>{record.result.prefillBottleneck}</dd></div>
            <div><dt>Decode Bottleneck</dt><dd>{record.result.decodeBottleneck}</dd></div>
            <div>
              <dt>Memory Fit</dt>
              <dd className={record.result.memoryFitsCapacity ? "history-fit history-fit--yes" : "history-fit history-fit--no"}>
                {record.result.memoryFitsCapacity ? "Fits" : "Exceeds capacity"}
              </dd>
            </div>
          </dl>
        </section>
        <div className="history-record__actions">
          <button
            type="button"
            className="danger-button"
            onClick={() => onDelete(record)}
          >
            删除此记录
          </button>
        </div>
      </div>
    </details>
  );
}

export function HistoryPage() {
  const { historyRecords, clearHistory, deleteHistoryRecord } = useCalculatorContext();
  const [modelFilter, setModelFilter] = useState("all");
  const [timeOrder, setTimeOrder] = useState<HistoryTimeOrder>("newest");

  const modelOptions = useMemo(
    () =>
      Array.from(
        new Map(
          historyRecords.map((record) => [
            record.modelId,
            record.modelDisplayName
          ])
        )
      ).sort((left, right) => left[1].localeCompare(right[1])),
    [historyRecords]
  );

  const visibleRecords = useMemo(() => {
    const filtered =
      modelFilter === "all"
        ? historyRecords
        : historyRecords.filter((record) => record.modelId === modelFilter);
    return [...filtered].sort((left, right) => {
      const delta =
        new Date(left.createdAt).getTime() - new Date(right.createdAt).getTime();
      return timeOrder === "oldest" ? delta : -delta;
    });
  }, [historyRecords, modelFilter, timeOrder]);

  function handleClearHistory() {
    if (historyRecords.length === 0) {
      return;
    }

    if (window.confirm("确定清空全部计算历史吗？此操作无法撤销。")) {
      clearHistory();
      setModelFilter("all");
    }
  }

  function handleDeleteRecord(record: CalculationHistoryRecord) {
    if (
      window.confirm(
        `确定删除 ${formatDate(record.createdAt)} 的 ${record.modelDisplayName} 计算记录吗？此操作无法撤销。`
      )
    ) {
      deleteHistoryRecord(record.id);
    }
  }

  return (
    <section className="page-section">
      <div className="page-heading">
        <div>
          <p className="eyebrow">Calculation Archive</p>
          <h2>历史记录</h2>
        </div>
        <p className="page-description">
          保存每次计算的模型、参数快照与结果摘要，数据仅存储在当前浏览器或桌面应用中。
        </p>
      </div>

      <article className="panel history-toolbar">
        <div className="history-toolbar__filters">
          <label className="field">
            <span>模型筛选</span>
            <select value={modelFilter} onChange={(event) => setModelFilter(event.target.value)}>
              <option value="all">全部模型</option>
              {modelOptions.map(([modelId, displayName]) => (
                <option key={modelId} value={modelId}>{displayName}</option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>时间顺序</span>
            <select
              value={timeOrder}
              onChange={(event) => setTimeOrder(event.target.value as HistoryTimeOrder)}
            >
              <option value="newest">最新优先</option>
              <option value="oldest">最早优先</option>
            </select>
          </label>
        </div>
        <div className="history-toolbar__actions">
          <span>显示 {visibleRecords.length} / {historyRecords.length} 条</span>
          <button
            type="button"
            className="danger-button"
            disabled={historyRecords.length === 0}
            onClick={handleClearHistory}
          >
            清空记录
          </button>
        </div>
      </article>

      {visibleRecords.length > 0 ? (
        <div className="history-list">
          {visibleRecords.map((record) => (
            <HistoryRecordCard
              key={record.id}
              record={record}
              onDelete={handleDeleteRecord}
            />
          ))}
        </div>
      ) : (
        <article className="panel panel--large history-empty">
          <h3>{historyRecords.length === 0 ? "暂无计算记录" : "没有符合筛选条件的记录"}</h3>
          <p>
            {historyRecords.length === 0
              ? "前往“性能计算”页面点击“计算性能”，成功计算后会自动保存记录。"
              : "请选择其他模型或切换到“全部模型”。"}
          </p>
        </article>
      )}
    </section>
  );
}
