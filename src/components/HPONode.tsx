import React from "react";
import { PredictionResult } from "@/utils/PredictionResult";
import { GroupedDistributionChart } from "@/components/GroupedDistributionChart";

interface HPONodeProps {
  node: PredictionResult;
  isExpanded?: boolean;
  onSelect: (node: PredictionResult | null) => void;
  onToggleExpand?: () => void;
  isCompleted?: boolean;
}

export const HPONode = ({
  node,
  isExpanded = false,
  onSelect,
  onToggleExpand,
  isCompleted = node.probability > 0,
}: HPONodeProps) => {
  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onSelect(isExpanded ? null : node);
    if (onToggleExpand) {
      onToggleExpand();
    }
  };

  const stateClass = !isCompleted
    ? "pending"
    : node.isPositive
      ? "positive"
      : "negative";

  return (
    <>
      <style jsx global>{`
        .hpo-node {
          position: relative;
          margin: 0.625rem 0;
          border-radius: 18px;
          border: 1px solid rgba(15, 23, 42, 0.08);
          background: rgba(255, 255, 255, 0.82);
          box-shadow:
            0 1px 2px rgba(15, 23, 42, 0.04),
            0 10px 24px rgba(15, 23, 42, 0.06);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          transition:
            transform 180ms ease,
            box-shadow 180ms ease,
            border-color 180ms ease,
            background-color 180ms ease;
          cursor: pointer;
          overflow: hidden;
        }

        .hpo-node:hover {
          transform: translateY(-1px);
          box-shadow:
            0 4px 10px rgba(15, 23, 42, 0.06),
            0 16px 32px rgba(15, 23, 42, 0.10);
        }

        .hpo-node.expanded {
          border-color: rgba(59, 130, 246, 0.22);
          box-shadow:
            0 6px 18px rgba(59, 130, 246, 0.10),
            0 18px 40px rgba(15, 23, 42, 0.10);
        }

        .hpo-node::before {
          content: "";
          position: absolute;
          inset: 0 auto 0 0;
          width: 4px;
          border-radius: 999px;
          background: transparent;
        }

        .hpo-node.positive::before {
          background: linear-gradient(180deg, #22c55e, #16a34a);
        }

        .hpo-node.negative::before {
          background: linear-gradient(180deg, #f97316, #ef4444);
        }

        .hpo-node.pending::before {
          background: linear-gradient(180deg, #94a3b8, #64748b);
        }

        .hpo-node-shell {
          padding: 1rem 1rem 0.95rem 1rem;
        }

        .hpo-node-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.75rem;
        }

        .hpo-node-main {
          min-width: 0;
          display: flex;
          align-items: center;
          gap: 0.75rem;
          flex: 1 1 auto;
        }

        .hpo-node-icon {
          width: 2rem;
          height: 2rem;
          border-radius: 999px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          flex: 0 0 auto;
          font-size: 0.9rem;
          border: 1px solid transparent;
          background: rgba(148, 163, 184, 0.12);
          color: #64748b;
        }

        .hpo-node.positive .hpo-node-icon {
          background: rgba(34, 197, 94, 0.12);
          color: #15803d;
          border-color: rgba(34, 197, 94, 0.18);
        }

        .hpo-node.negative .hpo-node-icon {
          background: rgba(239, 68, 68, 0.10);
          color: #dc2626;
          border-color: rgba(239, 68, 68, 0.16);
        }

        .hpo-node.pending .hpo-node-icon {
          background: rgba(148, 163, 184, 0.14);
          color: #64748b;
          border-color: rgba(148, 163, 184, 0.2);
        }

        .hpo-node-title-wrap {
          min-width: 0;
          flex: 1 1 auto;
        }

        .hpo-node-title {
          margin: 0;
          color: #0f172a;
          font-size: 0.97rem;
          font-weight: 700;
          line-height: 1.25;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .hpo-node-subtitle {
          margin-top: 0.2rem;
          color: #64748b;
          font-size: 0.78rem;
          line-height: 1.2;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .hpo-node-badge {
          flex: 0 0 auto;
          display: inline-flex;
          align-items: center;
          gap: 0.35rem;
          padding: 0.45rem 0.7rem;
          border-radius: 999px;
          font-size: 0.78rem;
          font-weight: 700;
          letter-spacing: 0.01em;
          border: 1px solid transparent;
          box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }

        .hpo-node-badge.positive {
          color: #166534;
          background: rgba(34, 197, 94, 0.14);
          border-color: rgba(34, 197, 94, 0.2);
        }

        .hpo-node-badge.negative {
          color: #b91c1c;
          background: rgba(239, 68, 68, 0.12);
          border-color: rgba(239, 68, 68, 0.18);
        }

        .hpo-node-badge.pending {
          color: #475569;
          background: rgba(241, 245, 249, 0.95);
          border-color: rgba(148, 163, 184, 0.2);
        }

        .hpo-node-progress {
          margin-top: 0.85rem;
          height: 8px;
          width: 100%;
          border-radius: 999px;
          overflow: hidden;
          background: rgba(226, 232, 240, 0.85);
        }

        .hpo-node-progress-fill {
          height: 100%;
          border-radius: inherit;
          transition: width 300ms ease;
        }

        .hpo-node-progress-fill.positive {
          background: linear-gradient(90deg, #22c55e, #10b981);
        }

        .hpo-node-progress-fill.negative {
          background: linear-gradient(90deg, #fb7185, #ef4444);
        }

        .hpo-node-progress-fill.pending {
          background: linear-gradient(90deg, #cbd5e1, #94a3b8);
        }

        .hpo-node-details {
          margin-top: 2rem;
          //padding: 1rem;
          //border-radius: 14px;
          //border: 1px solid rgba(148, 163, 184, 0.18);
          //background:
          //  linear-gradient(180deg, rgba(255,255,255,0.92), rgba(248,250,252,0.96));
          color: #0f172a;
        }

        .hpo-node-section-title {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.75rem;
          padding-bottom: 0.45rem;
          border-bottom: 1px solid rgba(148, 163, 184, 0.18);
          font-size: 0.78rem;
          font-weight: 800;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          color: #334155;
        }

        .hpo-node-meta-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          align-items: center;
          margin-bottom: 0.6rem;
        }

        .hpo-node-chip {
          display: inline-flex;
          align-items: center;
          gap: 0.35rem;
          padding: 0.3rem 0.6rem;
          border-radius: 999px;
          border: 1px solid rgba(148, 163, 184, 0.18);
          background: rgba(248, 250, 252, 0.9);
          color: #334155;
          font-size: 0.74rem;
          font-weight: 600;
        }

        .hpo-node-description {
          color: #475569;
          line-height: 1.55;
        }

        .hpo-metric-card {
          height: 100%;
          padding: 0.75rem;
          border-radius: 14px;
          border: 1px solid rgba(148, 163, 184, 0.14);
          background: rgba(248, 250, 252, 0.92);
          text-align: center;
        }

        .hpo-metric-label {
          color: #64748b;
          font-size: 0.68rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }

        .hpo-metric-value {
          margin-top: 0.3rem;
          color: #0f172a;
          font-size: 0.95rem;
          font-weight: 800;
        }

        .hpo-confusion-grid {
          max-width: 240px;
          border-radius: 16px;
          overflow: hidden;
          border: 1px solid rgba(148, 163, 184, 0.16);
          background: #fff;
        }

        .hpo-confusion-cell {
          padding: 0.7rem 0.55rem;
          text-align: center;
        }

        .hpo-confusion-cell.positive {
          background: rgba(34, 197, 94, 0.10);
        }

        .hpo-confusion-cell.negative {
          background: rgba(239, 68, 68, 0.08);
        }

        .hpo-confusion-label {
          color: #64748b;
          font-size: 0.64rem;
          line-height: 1.15;
        }

        .hpo-confusion-value {
          margin-top: 0.2rem;
          color: #0f172a;
          font-size: 0.82rem;
          font-weight: 800;
        }

        .hpo-syndrome-badge {
          display: inline-flex;
          align-items: center;
          padding: 0.35rem 0.55rem;
          border-radius: 999px;
          border: 1px solid rgba(148, 163, 184, 0.18);
          background: rgba(248, 250, 252, 0.9);
          color: #334155;
          font-size: 0.68rem;
          font-weight: 600;
        }
      `}</style>

      <div
        className={`hpo-node ${stateClass} ${isExpanded ? "expanded" : ""}`}
        onClick={handleClick}
      >
        <div className="hpo-node-shell">
          <div className="hpo-node-header">
            <div className="hpo-node-main">
            <span className={`hpo-node-badge ${stateClass}`}>
              {isCompleted ? `${(node.probability * 100).toFixed(1)}%` : "Pending"}
            </span>

              <div className="hpo-node-title-wrap">
                <div className="hpo-node-title">{node.name}</div>
                {/*<div className="hpo-node-subtitle">*/}
                {/*  {node.hpoId || node.id}*/}
                {/*</div>*/}
              </div>
            </div>

          </div>

          {/*<div className="hpo-node-progress" aria-hidden="true">*/}
          {/*  <div*/}
          {/*    className={`hpo-node-progress-fill ${stateClass}`}*/}
          {/*    style={{ width: `${isCompleted ? node.probability * 100 : 12}%` }}*/}
          {/*  />*/}
          {/*</div>*/}

          {isExpanded && (
            <div className="hpo-node-details">
              <div className="mb-3">
                {/*<div className="hpo-node-section-title">Overview</div>*/}

                <div className="hpo-node-meta-row">
                  <span className="hpo-node-chip">
                    <strong>ID:</strong> {node.hpoId || node.id}
                  </span>
                  {/*{node.metrics?.validation_samples !== undefined && (*/}
                  {/*  <span className="hpo-node-chip">*/}
                  {/*    <strong>Validation Samples:</strong> {node.metrics.validation_samples}*/}
                  {/*  </span>*/}
                  {/*)}*/}
                </div>

                {node.description && (
                  <div className="hpo-node-description small">
                    {node.description}
                  </div>
                )}
              </div>

              {node.metrics && (
                <div className="mt-3">
                  <div className="hpo-node-section-title">Performance metrics</div>

                  <div className="row g-2 text-center">
                    <div className="col-6">
                      <div className="hpo-metric-card">
                        <div className="hpo-metric-label">Sensitivity</div>
                        <div className="hpo-metric-value">
                          {(node.metrics.recall * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                    <div className="col-6">
                      <div className="hpo-metric-card">
                        <div className="hpo-metric-label">Specificity</div>
                        <div className="hpo-metric-value">
                          {(node.metrics.specificity * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="mt-3">
                    <div
                      className="small text-muted mb-2 fw-bold"
                      style={{ fontSize: "0.75rem" }}
                    >
                      Confusion Matrix
                    </div>

                    <div className="row g-0 hpo-confusion-grid">
                      <div className="col-6 hpo-confusion-cell positive border-end border-bottom">
                        <div className="hpo-confusion-label">True Positive</div>
                        <div className="hpo-confusion-value">
                          {((node.metrics.tp / (node.metrics.tp + node.metrics.fn)) * 100).toFixed(1)}%
                        </div>
                      </div>

                      <div className="col-6 hpo-confusion-cell negative border-bottom">
                        <div className="hpo-confusion-label">False Positive</div>
                        <div className="hpo-confusion-value">
                          {((node.metrics.fp / (node.metrics.fp + node.metrics.tn)) * 100).toFixed(1)}%
                        </div>
                      </div>

                      <div className="col-6 hpo-confusion-cell negative border-end">
                        <div className="hpo-confusion-label">False Negative</div>
                        <div className="hpo-confusion-value">
                          {((node.metrics.fn / (node.metrics.tp + node.metrics.fn)) * 100).toFixed(1)}%
                        </div>
                      </div>

                      <div className="col-6 hpo-confusion-cell positive">
                        <div className="hpo-confusion-label">True Negative</div>
                        <div className="hpo-confusion-value">
                          {((node.metrics.tn / (node.metrics.fp + node.metrics.tn)) * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {node.database?.affected && node.database?.unaffected && (
                <div className="mt-3">
                  <div className="hpo-node-section-title">Training data distribution</div>

                  <div className="row g-3">
                    <div className="col-lg-3 col-md-6">
                      <GroupedDistributionChart
                        title="Gender"
                        categories={node.database.affected.gender}
                        affected={node.database.affected.gender}
                        unaffected={node.database.unaffected.gender}
                      />
                    </div>

                    <div className="col-lg-3 col-md-6">
                      <GroupedDistributionChart
                        title="Ethnicity"
                        categories={node.database.affected.ethnicity}
                        affected={node.database.affected.ethnicity}
                        unaffected={node.database.unaffected.ethnicity}
                      />
                    </div>

                    <div className="col-lg-3 col-md-6">
                      <GroupedDistributionChart
                        title="Age Groups"
                        categories={node.database.affected.ageGroups}
                        affected={node.database.affected.ageGroups}
                        unaffected={node.database.unaffected.ageGroups}
                      />
                    </div>

                    <div className="col-lg-3 col-md-6">
                      <GroupedDistributionChart
                        title="Sources"
                        categories={node.database.affected.databaseSource}
                        affected={node.database.affected.databaseSource}
                        unaffected={node.database.unaffected.databaseSource}
                      />
                    </div>
                  </div>
                </div>
              )}

              {node.syndromes && node.syndromes.length > 0 && (
                <div className="mt-3">
                  <div className="hpo-node-section-title">Syndromes in data distribution</div>
                  <div className="d-flex flex-wrap gap-2">
                    {node.syndromes.map((s, idx) => (
                      <span key={idx} className="hpo-syndrome-badge">
                        {s.name} ({s.count})
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
};