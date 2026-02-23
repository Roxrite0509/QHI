import { useState, useEffect } from "react";

const COLORS = {
  bg: "#060e1a",
  card: "#0a1728",
  border: "#0f2847",
  blue: "#2d7dd2",
  blue2: "#5ba4e8",
  teal: "#00c9a7",
  purple: "#9b6dff",
  amber: "#f0a500",
  green: "#22c55e",
  red: "#ef4444",
  muted: "#5a7a9e",
  text: "#dce8f5",
};

const useVisible = (delay = 0) => {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setVisible(true), delay);
    return () => clearTimeout(t);
  }, []);
  return visible;
};

const FadeIn = ({ children, delay = 0, style = {} }) => {
  const vis = useVisible(delay);
  return (
    <div style={{
      opacity: vis ? 1 : 0,
      transform: vis ? "translateY(0)" : "translateY(18px)",
      transition: "opacity 0.55s ease, transform 0.55s ease",
      ...style
    }}>
      {children}
    </div>
  );
};

const Arrow = ({ color = COLORS.blue }) => (
  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", height: 44 }}>
    <div style={{ width: 2, flex: 1, background: `linear-gradient(to bottom, ${color}44, ${color})` }} />
    <div style={{
      width: 0, height: 0,
      borderLeft: "7px solid transparent",
      borderRight: "7px solid transparent",
      borderTop: `9px solid ${color}`,
    }} />
  </div>
);

const SectionLabel = ({ step, title, color = COLORS.blue }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
    <div style={{
      fontFamily: "monospace", fontSize: 10, fontWeight: 700,
      background: `${color}22`, color, border: `1px solid ${color}44`,
      padding: "3px 9px", borderRadius: 20, letterSpacing: 2, textTransform: "uppercase"
    }}>{step}</div>
    <span style={{ fontFamily: "monospace", fontSize: 11, color: COLORS.muted, letterSpacing: 1 }}>{title}</span>
  </div>
);

const Card = ({ children, accent, style = {} }) => (
  <div style={{
    background: COLORS.card,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 14,
    overflow: "hidden",
    position: "relative",
    ...style
  }}>
    {accent && <div style={{ height: 3, background: accent }} />}
    {children}
  </div>
);

const Tag = ({ children, color }) => (
  <span style={{
    fontFamily: "monospace", fontSize: 10, fontWeight: 600,
    padding: "3px 9px", borderRadius: 5,
    background: `${color}18`, color, border: `1px solid ${color}30`,
    display: "inline-block",
  }}>{children}</span>
);

const Token = ({ children, highlight }) => (
  <span style={{
    fontFamily: "monospace", fontSize: 10,
    padding: "3px 9px", borderRadius: 20,
    background: highlight ? `${COLORS.teal}18` : `${COLORS.blue}12`,
    color: highlight ? COLORS.teal : COLORS.blue2,
    border: `1px solid ${highlight ? COLORS.teal + "35" : COLORS.blue + "30"}`,
    fontWeight: highlight ? 700 : 400,
  }}>{children}</span>
);

export default function App() {
  const [hovered, setHovered] = useState(null);

  const probeDefs = [
    {
      id: "C", icon: "◎", color: COLORS.purple,
      name: "Probe-C", arch: "Logistic Regression · L2",
      output: "uncertainty ∈ [0, 1]",
      desc: "Detects model uncertainty from hidden state geometry. Linear probe on middle-layer activations.",
    },
    {
      id: "R", icon: "⊕", color: COLORS.blue2,
      name: "Probe-R", arch: "MLP (256 → 64 → 32 → 5) · ReLU",
      output: "risk_score ∈ [1, 5]",
      desc: "ICD-10 aligned 5-class clinical risk. MLP captures joint drug × dose × domain interactions.",
    },
    {
      id: "V", icon: "⊗", color: COLORS.teal,
      name: "Probe-V", arch: "L1 Logistic · Sparse (C=0.5)",
      output: "violation_prob ∈ [0, 1]",
      desc: "Causal/factual contradiction detection. L1 sparsity enables interpretable weight auditing.",
    },
  ];

  const gates = [
    { range: "QHI < 5", name: "AUTO_USE", icon: "✓", color: COLORS.green, iso: "Acceptable", desc: "Deploy safely without human review." },
    { range: "5 – 19.99", name: "REVIEW", icon: "⚠", color: COLORS.amber, iso: "ALARP", desc: "Flag for clinician verification before use." },
    { range: "QHI ≥ 20", name: "BLOCK", icon: "✕", color: COLORS.red, iso: "Unacceptable", desc: "Reject output — escalate to expert." },
  ];

  return (
    <div style={{
      background: COLORS.bg, minHeight: "100vh",
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      color: COLORS.text, padding: "36px 20px 60px",
      backgroundImage: "radial-gradient(ellipse at 20% 0%, #0d2a5010 0%, transparent 60%), radial-gradient(ellipse at 80% 100%, #00c9a708 0%, transparent 60%)",
    }}>
      <div style={{ maxWidth: 820, margin: "0 auto" }}>

        {/* ── HEADER ─────────────────────────────────────────── */}
        <FadeIn delay={0}>
          <div style={{ textAlign: "center", marginBottom: 44 }}>
            <div style={{
              display: "inline-block", fontFamily: "monospace", fontSize: 10,
              letterSpacing: 3, textTransform: "uppercase", color: COLORS.teal,
              border: `1px solid ${COLORS.teal}40`, padding: "5px 16px",
              borderRadius: 20, background: `${COLORS.teal}0a`, marginBottom: 16,
            }}>System Architecture · QHI-Probe</div>
            <h1 style={{
              fontSize: "clamp(26px,5vw,40px)", fontWeight: 800,
              letterSpacing: -1, color: "#fff", lineHeight: 1.1, marginBottom: 10,
            }}>
              Clinical LLM <span style={{ color: COLORS.blue2 }}>Hallucination</span> Pipeline
            </h1>
            <p style={{ fontFamily: "monospace", fontSize: 12, color: COLORS.muted }}>
              Quantified Hallucination Index · Sparse Entity Probing · ISO 14971 Gates
            </p>
          </div>
        </FadeIn>

        {/* ── INPUT NODE ─────────────────────────────────────── */}
        <FadeIn delay={80}>
          <Card accent={`linear-gradient(90deg,${COLORS.blue},${COLORS.blue2})`}>
            <div style={{ padding: "20px 26px 22px" }}>
              <SectionLabel step="INPUT" title="Clinical LLM Output Text" color={COLORS.blue} />
              <div style={{ fontFamily: "monospace", fontSize: 12, color: COLORS.muted, marginBottom: 14 }}>
                Any response from ChatGPT · Gemini · Claude · BioMedLM
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {[
                  ["Patient presents with", false], ["STEMI", true],
                  ["crushing chest pain", false], ["aspirin 325mg", true],
                  ["for 90 minutes", false], ["cath lab", true],
                  ["activate", false], ["heparin", true], ["PCI", true],
                ].map(([t, h], i) => <Token key={i} highlight={h}>{t}</Token>)}
              </div>
              <div style={{
                marginTop: 14, fontFamily: "monospace", fontSize: 10,
                color: COLORS.teal, display: "flex", alignItems: "center", gap: 6
              }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: COLORS.teal, animation: "pulse 2s infinite" }} />
                <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}`}</style>
                Highlighted tokens = medical entities detected for sparse probing
              </div>
            </div>
          </Card>
        </FadeIn>

        <FadeIn delay={150}><Arrow color={COLORS.blue} /></FadeIn>

        {/* ── STAGE 1: ENTITY EXTRACTION ─────────────────────── */}
        <FadeIn delay={200}>
          <Card accent={`linear-gradient(90deg,${COLORS.blue},${COLORS.blue2}80)`}>
            <div style={{ padding: "20px 26px 22px" }}>
              <SectionLabel step="Stage 1" title="Entity Extraction" color={COLORS.blue} />
              <div style={{ fontFamily: "monospace", fontSize: 12, color: COLORS.text, fontWeight: 600, marginBottom: 6 }}>
                NER via scispaCy / BioNER → Medical Entity Tokens (e₁, e₂, …, eₖ)
              </div>
              <div style={{ fontFamily: "monospace", fontSize: 11, color: COLORS.muted, lineHeight: 1.8 }}>
                <span style={{ color: COLORS.blue2 }}>k ≈ 5–15 entity tokens</span> selected from N ≈ 200–500 total tokens
                &nbsp;·&nbsp; Reduces feature computation by <span style={{ color: COLORS.amber }}>93–97%</span>
              </div>
              <div style={{ marginTop: 14, display: "flex", gap: 8, flexWrap: "wrap" }}>
                {["STEMI","aspirin 325mg","cath lab","heparin","PCI"].map(t => <Token key={t} highlight>{t}</Token>)}
                <span style={{ fontFamily: "monospace", fontSize: 10, color: COLORS.muted, alignSelf: "center" }}>← extracted entities only</span>
              </div>
            </div>
          </Card>
        </FadeIn>

        <FadeIn delay={260}><Arrow color={COLORS.teal} /></FadeIn>

        {/* ── STAGE 2: FROZEN LLM ────────────────────────────── */}
        <FadeIn delay={310}>
          <Card accent={`linear-gradient(90deg,${COLORS.teal},${COLORS.blue2})`}>
            <div style={{ padding: "20px 26px 22px" }}>
              <SectionLabel step="Stage 2" title="Frozen LLM Backbone" color={COLORS.teal} />
              <div style={{ fontFamily: "monospace", fontSize: 12, color: COLORS.text, fontWeight: 600, marginBottom: 10 }}>
                Forward pass: <span style={{ color: COLORS.red }}>NO GRADIENT</span> · torch.no_grad() · model.eval()
              </div>
              <div style={{ fontFamily: "monospace", fontSize: 11, color: COLORS.muted, lineHeight: 1.9 }}>
                Extract <span style={{ color: COLORS.teal }}>hidden_states[L8, L16, L24]</span> at entity positions <span style={{ color: COLORS.teal, fontWeight: 700 }}>ONLY</span><br />
                Weighted sum: <span style={{ color: COLORS.teal }}>h = 0.2·h₈ + 0.5·h₁₆ + 0.3·h₂₄</span>  (dim = 768)<br />
                Project to 256-dim: <span style={{ color: COLORS.teal }}>h′ = W·h</span>  (learned projection matrix)
              </div>
              <div style={{ marginTop: 14, display: "flex", gap: 8, flexWrap: "wrap" }}>
                {[
                  { label: "Layer 8 — Syntactic", weight: "0.2×", emphasis: false },
                  { label: "Layer 16 — Factual", weight: "0.5×", emphasis: true },
                  { label: "Layer 24 — Semantic", weight: "0.3×", emphasis: false },
                ].map(({ label, weight, emphasis }) => (
                  <div key={label} style={{
                    fontFamily: "monospace", fontSize: 10, padding: "5px 12px",
                    borderRadius: 7, border: `1px solid ${COLORS.teal}${emphasis ? "50" : "25"}`,
                    background: `${COLORS.teal}${emphasis ? "20" : "0a"}`,
                    color: emphasis ? COLORS.teal : COLORS.muted,
                    fontWeight: emphasis ? 700 : 400,
                  }}>
                    {label} · <span style={{ color: COLORS.teal }}>{weight}</span>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </FadeIn>

        <FadeIn delay={370}><Arrow color={COLORS.purple} /></FadeIn>

        {/* ── STAGE 3: THREE PROBES ──────────────────────────── */}
        <FadeIn delay={420}>
          <div style={{
            border: `1px solid ${COLORS.border}`, borderRadius: 14,
            overflow: "hidden", background: COLORS.card,
          }}>
            {/* Header */}
            <div style={{
              borderBottom: `1px solid ${COLORS.border}`,
              padding: "16px 26px 14px",
              background: `linear-gradient(90deg,${COLORS.purple}12,${COLORS.teal}08)`,
              position: "relative", overflow: "hidden",
            }}>
              <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 3, background: `linear-gradient(90deg,${COLORS.purple},${COLORS.blue2},${COLORS.teal})` }} />
              <SectionLabel step="Stage 3 · Parallel" title="Three Lightweight Probes — h′ ∈ ℝ²⁵⁶ → three signals" color={COLORS.purple} />
              <div style={{ fontFamily: "monospace", fontSize: 10, color: COLORS.muted }}>
                Total probe parameters: &lt; 500K · Each runs in &lt; 0.2ms on CPU · No GPU required
              </div>
            </div>

            {/* Probes Grid */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)" }}>
              {probeDefs.map((p, i) => (
                <div
                  key={p.id}
                  onMouseEnter={() => setHovered(p.id)}
                  onMouseLeave={() => setHovered(null)}
                  style={{
                    padding: "20px 18px 22px",
                    borderRight: i < 2 ? `1px solid ${COLORS.border}` : "none",
                    background: hovered === p.id ? `${p.color}08` : "transparent",
                    transition: "background 0.2s", cursor: "default",
                  }}
                >
                  <div style={{
                    width: 40, height: 40, borderRadius: 10, marginBottom: 12,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 20, background: `${p.color}18`, border: `1px solid ${p.color}30`,
                    color: p.color, fontFamily: "monospace",
                  }}>{p.icon}</div>
                  <div style={{ fontFamily: "monospace", fontSize: 13, fontWeight: 700, color: "#fff", marginBottom: 3 }}>{p.name}</div>
                  <div style={{ fontFamily: "monospace", fontSize: 9, color: COLORS.muted, marginBottom: 10, lineHeight: 1.5 }}>{p.arch}</div>
                  <div style={{
                    fontFamily: "monospace", fontSize: 10, fontWeight: 700,
                    padding: "5px 10px", borderRadius: 6, marginBottom: 10,
                    background: `${p.color}15`, color: p.color, border: `1px solid ${p.color}30`,
                  }}>{p.output}</div>
                  <div style={{ fontFamily: "monospace", fontSize: 10, color: COLORS.muted, lineHeight: 1.7 }}>{p.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>

        <FadeIn delay={490}><Arrow color={COLORS.amber} /></FadeIn>

        {/* ── FORMULA ────────────────────────────────────────── */}
        <FadeIn delay={540}>
          <Card accent={`linear-gradient(90deg,${COLORS.amber},#f97316,#eab308)`}>
            <div style={{ padding: "22px 26px 24px" }}>
              <SectionLabel step="Stage 4" title="Score Computation" color={COLORS.amber} />

              {/* Formula */}
              <div style={{
                fontFamily: "monospace", fontSize: "clamp(13px,2.5vw,19px)", fontWeight: 700,
                textAlign: "center", padding: "16px 0",
                borderTop: `1px solid ${COLORS.amber}20`, borderBottom: `1px solid ${COLORS.amber}20`,
                marginBottom: 14, letterSpacing: 1,
                background: `linear-gradient(135deg,${COLORS.amber}08,transparent)`,
                borderRadius: 8,
              }}>
                <span style={{ color: COLORS.amber }}>QHI</span>
                <span style={{ color: COLORS.muted }}> = </span>
                <span style={{ color: COLORS.purple }}>uncertainty</span>
                <span style={{ color: COLORS.muted }}> × </span>
                <span style={{ color: COLORS.blue2 }}>risk_score</span>
                <span style={{ color: COLORS.muted }}> × </span>
                <span style={{ color: COLORS.teal }}>violation_prob</span>
                <span style={{ color: COLORS.muted }}> × </span>
                <span style={{ color: COLORS.amber }}>5</span>
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 10 }}>
                <div style={{ fontFamily: "monospace", fontSize: 11, color: COLORS.muted }}>
                  Range: <span style={{ color: COLORS.amber, fontWeight: 700 }}>0</span> (no hallucination)
                  &nbsp;→&nbsp;
                  <span style={{ color: COLORS.amber, fontWeight: 700 }}>25</span> (critical hallucination)
                </div>
                <div style={{ fontFamily: "monospace", fontSize: 10, color: COLORS.muted }}>
                  All 3 signals must be elevated simultaneously → prevents false alarms
                </div>
              </div>
            </div>
          </Card>
        </FadeIn>

        <FadeIn delay={600}><Arrow color={COLORS.green} /></FadeIn>

        {/* ── GATES ──────────────────────────────────────────── */}
        <FadeIn delay={650}>
          <div style={{
            border: `1px solid ${COLORS.border}`, borderRadius: 14,
            overflow: "hidden", background: COLORS.card,
          }}>
            {/* Header */}
            <div style={{
              borderBottom: `1px solid ${COLORS.border}`,
              padding: "16px 26px 14px",
              position: "relative", overflow: "hidden",
            }}>
              <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 3, background: `linear-gradient(90deg,${COLORS.green},${COLORS.amber},${COLORS.red})` }} />
              <SectionLabel step="Stage 5 · Output" title="Operational Safety Gate · ISO 14971 Aligned" color={COLORS.green} />
              <div style={{ fontFamily: "monospace", fontSize: 10, color: COLORS.muted }}>Clinical deployment decision — auditable output for regulatory compliance</div>
            </div>

            {/* Gates Grid */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)" }}>
              {gates.map((g, i) => (
                <div key={g.name} style={{
                  padding: "20px 18px 22px",
                  borderRight: i < 2 ? `1px solid ${COLORS.border}` : "none",
                  background: `${g.color}06`,
                }}>
                  <div style={{
                    width: 44, height: 44, borderRadius: 12, marginBottom: 12,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 20, fontWeight: 700,
                    background: `${g.color}15`, border: `1px solid ${g.color}30`,
                    color: g.color,
                  }}>{g.icon}</div>
                  <div style={{ fontFamily: "monospace", fontSize: 12, fontWeight: 700, color: g.color, marginBottom: 3 }}>{g.range}</div>
                  <div style={{ fontSize: 15, fontWeight: 800, color: g.color, marginBottom: 10 }}>{g.name}</div>
                  <div style={{ fontFamily: "monospace", fontSize: 10, color: COLORS.muted, lineHeight: 1.7, marginBottom: 10 }}>{g.desc}</div>
                  <div style={{
                    display: "inline-block", fontFamily: "monospace", fontSize: 9,
                    padding: "3px 8px", borderRadius: 4, letterSpacing: 0.5,
                    background: `${g.color}12`, color: g.color, border: `1px solid ${g.color}25`,
                  }}>ISO 14971 · {g.iso}</div>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>

        {/* ── LEGEND ─────────────────────────────────────────── */}
        <FadeIn delay={780}>
          <div style={{
            marginTop: 36, padding: "18px 24px",
            border: `1px solid ${COLORS.border}`, borderRadius: 12,
            background: COLORS.card, display: "flex", flexWrap: "wrap", gap: "10px 28px",
          }}>
            <div style={{ fontFamily: "monospace", fontSize: 9, letterSpacing: 2.5, textTransform: "uppercase", color: COLORS.muted, width: "100%", marginBottom: 4 }}>Component Legend</div>
            {[
              [COLORS.purple, "Probe-C · Uncertainty"],
              [COLORS.blue2,  "Probe-R · Risk Score"],
              [COLORS.teal,   "Probe-V · Violation Probability"],
              [COLORS.amber,  "QHI Formula · 0–25 Scale"],
              [COLORS.green,  "AUTO_USE Gate"],
              [COLORS.amber,  "REVIEW Gate"],
              [COLORS.red,    "BLOCK Gate"],
            ].map(([c, label]) => (
              <div key={label} style={{ display: "flex", alignItems: "center", gap: 7, fontFamily: "monospace", fontSize: 10, color: COLORS.muted }}>
                <div style={{ width: 7, height: 7, borderRadius: "50%", background: c, flexShrink: 0 }} />
                {label}
              </div>
            ))}
            <div style={{ fontFamily: "monospace", fontSize: 9, color: COLORS.muted, width: "100%", marginTop: 6, paddingTop: 10, borderTop: `1px solid ${COLORS.border}` }}>
              ⚡ Inference: &lt;1ms CPU · No GPU required · &lt;500K probe parameters · ISO 14971 compliant outputs
            </div>
          </div>
        </FadeIn>

      </div>
    </div>
  );
}
