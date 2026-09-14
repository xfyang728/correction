import { MaskReveal } from "../../components/MaskReveal";
import type { ChapterStepProps } from "../../registry/types";
import "./TheProblem.css";

export default function TheProblemChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="tp-scene scene-pad">
        <header className="masthead">
          <span className="brand">默写批改台</span>
          <span className="issue">Chapter 03 · 坐标全偏了</span>
        </header>
        <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

        <div className="tp-intro">
          <div className="kicker">第一版结果 · 尴尬了</div>
          <MaskReveal show duration={900}>
            <h2 className="tp-headline serif-cn">
              认字没问题，<span className="tp-em">坐标全偏了</span>
            </h2>
          </MaskReveal>
          <MaskReveal show delay={500} duration={800}>
            <p className="tp-sub serif-cn">像把字幕放错了位置</p>
          </MaskReveal>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="tp-scene scene-pad">
        <header className="masthead">
          <span className="brand">默写批改台</span>
          <span className="issue">Chapter 03 · 偏下 + 偏右</span>
        </header>
        <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

        <div className="tp-report-wrap">
          <div className="kicker">批注报告 · 旧版</div>
          <div className="tp-report">
            {/* handwriting traces (faint) */}
            <div className="tp-handwriting">
              <div className="tp-hand-line">随君直到夜郎西</div>
              <div className="tp-hand-line">海内存知己</div>
              <div className="tp-hand-line">天涯若比邻</div>
            </div>
            {/* annotations — offset down + right */}
            <div className="tp-annotations">
              <div className="tp-anno-line tp-anno-1">
                <span className="tp-anno-ch tp-anno-correct">随君直到夜郎西</span>
              </div>
              <div className="tp-anno-line tp-anno-2">
                <span className="tp-anno-ch tp-anno-correct">海内存知己</span>
              </div>
              <div className="tp-anno-line tp-anno-3">
                <span className="tp-anno-ch tp-anno-correct">天涯若比邻</span>
              </div>
            </div>
            {/* arrows */}
            <svg className="tp-arrows" viewBox="0 0 600 360" preserveAspectRatio="none">
              <defs>
                <marker id="tp-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
                  <path d="M 0 0 L 8 3 L 0 6 Z" fill="#f87171" />
                </marker>
              </defs>
              <path d="M 80 100 L 80 140" stroke="#f87171" strokeWidth="3" fill="none" markerEnd="url(#tp-arrow)" />
              <path d="M 80 180 L 80 220" stroke="#f87171" strokeWidth="3" fill="none" markerEnd="url(#tp-arrow)" />
              <path d="M 80 260 L 80 300" stroke="#f87171" strokeWidth="3" fill="none" markerEnd="url(#tp-arrow)" />
              <path d="M 110 100 L 70 100" stroke="#f87171" strokeWidth="3" fill="none" markerEnd="url(#tp-arrow)" />
            </svg>
          </div>
          <div className="tp-arrows-label">
            <span className="tp-arrow-label tp-arrow-down">↓ 偏下</span>
            <span className="tp-arrow-label tp-arrow-right">→ 首字偏右</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="tp-scene scene-pad">
      <header className="masthead">
        <span className="brand">默写批改台</span>
        <span className="issue">Chapter 03 · Y 坐标老毛病</span>
      </header>
      <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

      <div className="tp-yaxis">
        <div className="kicker">Qwen3-VL 的老毛病</div>
        <MaskReveal show duration={900}>
          <h2 className="tp-y-head serif-cn">
            好几道题共用 <span className="tp-em">同一个 Y 值</span>
          </h2>
        </MaskReveal>

        <div className="tp-y-diagram">
          {/* Y axis */}
          <div className="tp-y-axis">
            <span className="tp-y-tick">0.6</span>
            <span className="tp-y-tick">0.5</span>
            <span className="tp-y-tick">0.4</span>
            <span className="tp-y-tick">0.3</span>
          </div>

          {/* 3 questions all at same Y */}
          <div className="tp-y-questions">
            {[
              { label: "题1", y: 0.45 },
              { label: "题5", y: 0.48 },
              { label: "题7", y: 0.52 },
            ].map((q, i) => (
              <MaskReveal key={q.label} show delay={300 + i * 350} duration={700}>
                <div
                  className="tp-y-box"
                  style={{
                    top: `${(1 - q.y) * 100}%`,
                    left: `${i * 180 + 40}px`,
                  }}
                >
                  <span className="tp-y-box-label">{q.label}</span>
                  <span className="tp-y-box-val">y={q.y}</span>
                </div>
              </MaskReveal>
            ))}
            {/* chalk circle around the shared Y zone */}
            <svg className="tp-y-circle" viewBox="0 0 600 360" preserveAspectRatio="none">
              <ellipse
                cx="300"
                cy="180"
                rx="280"
                ry="50"
                fill="none"
                stroke="var(--accent)"
                strokeWidth="3"
                strokeDasharray="8 6"
                className="tp-circle-anim"
              />
            </svg>
          </div>
        </div>

        <MaskReveal show delay={1500} duration={800}>
          <p className="tp-y-foot serif-cn">
            实际它们在页面<span className="tp-em">完全不同的高度</span>
          </p>
        </MaskReveal>
      </div>
    </div>
  );
}
