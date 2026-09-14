import { MaskReveal } from "../../components/MaskReveal";
import type { ChapterStepProps } from "../../registry/types";
import "./XFix.css";

const CHARS = [
  { c: "(", w: 1, type: "ascii" },
  { c: "1", w: 1, type: "ascii" },
  { c: ")", w: 1, type: "ascii" },
  { c: " ", w: 1, type: "ascii" },
  { c: "随", w: 2, type: "cjk" },
  { c: "君", w: 2, type: "cjk" },
  { c: "直", w: 2, type: "cjk" },
  { c: "到", w: 2, type: "cjk" },
  { c: "夜", w: 2, type: "cjk" },
  { c: "郎", w: 2, type: "cjk" },
  { c: "西", w: 2, type: "cjk" },
];

export default function XFixChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="xf-scene scene-pad">
        <header className="masthead">
          <span className="brand">默写批改台</span>
          <span className="issue">Chapter 05 · 首字偏右</span>
        </header>
        <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

        <div className="xf-intro">
          <div className="kicker">X 偏移 · 90 ~ 130 像素</div>
          <MaskReveal show duration={900}>
            <h2 className="xf-head serif-cn">
              等宽假设 · <span className="xf-em">高估了 ASCII</span>
            </h2>
          </MaskReveal>
        </div>

        <div className="xf-grid-wrap">
          <div className="xf-grid-label">
            <span className="xf-grid-side label-mono">题号 (ASCII · 窄)</span>
            <span className="xf-grid-side label-mono xf-grid-side-cjk">答案 (中文 · 宽)</span>
          </div>
          <div className="xf-grid">
            {CHARS.map((ch, i) => (
              <MaskReveal key={i} show delay={i * 80} duration={500}>
                <div
                  className={`xf-cell ${ch.type === "cjk" ? "xf-cell-cjk" : "xf-cell-ascii"}`}
                  style={{ width: ch.w === 2 ? 80 : 40 }}
                >
                  {ch.c}
                </div>
              </MaskReveal>
            ))}
          </div>

          <div className="xf-marker-line">
            <div className="xf-marker xf-marker-wrong" style={{ left: "50%" }}>
              <span className="xf-marker-label">旧公式 · 随 起始位置</span>
              <span className="xf-marker-val">偏右 100px</span>
            </div>
          </div>
        </div>

        <MaskReveal show delay={1200} duration={800}>
          <p className="xf-foot serif-cn">
            ASCII 被当成中文宽度 → <span className="xf-em">答案起始被推到右边</span>
          </p>
        </MaskReveal>
      </div>
    );
  }

  return (
    <div className="xf-scene scene-pad">
      <header className="masthead">
        <span className="brand">默写批改台</span>
        <span className="issue">Chapter 05 · 权重修复</span>
      </header>
      <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

      <div className="xf-intro">
        <div className="kicker">east_asian_width · CJK=2 / ASCII=1</div>
        <MaskReveal show duration={900}>
          <h2 className="xf-head serif-cn">
            按真实宽度 <span className="xf-em">重新算位置</span>
          </h2>
        </MaskReveal>
      </div>

      <div className="xf-fix-grid">
        <div className="xf-grid">
          {CHARS.map((ch, i) => (
            <div
              key={i}
              className={`xf-cell ${ch.type === "cjk" ? "xf-cell-cjk" : "xf-cell-ascii"}`}
              style={{ width: ch.w === 2 ? 80 : 40 }}
            >
              {ch.c}
            </div>
          ))}
        </div>

        <div className="xf-marker-line">
          <MaskReveal show duration={800}>
            <div className="xf-marker xf-marker-old" style={{ left: "50%" }}>
              <span className="xf-marker-label">旧位置</span>
            </div>
          </MaskReveal>
          <MaskReveal show delay={600} duration={1000}>
            <div className="xf-marker xf-marker-new" style={{ left: "30%" }}>
              <span className="xf-marker-label">新位置</span>
              <span className="xf-marker-val">-100px</span>
            </div>
          </MaskReveal>
          {/* shift arrow */}
          <svg className="xf-shift" viewBox="0 0 200 24">
            <path d="M 160 12 L 40 12 M 50 4 L 38 12 L 50 20" />
          </svg>
        </div>
      </div>

      <MaskReveal show delay={1400} duration={800}>
        <p className="xf-foot serif-cn">
          首字左移 <span className="xf-em">~100 像素</span> · 正好对齐答案起始处
        </p>
      </MaskReveal>
    </div>
  );
}
