import { MaskReveal } from "../../components/MaskReveal";
import type { ChapterStepProps } from "../../registry/types";
import "./Polish.css";

export default function PolishChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="po-scene scene-pad">
        <header className="masthead">
          <span className="brand">默写批改台</span>
          <span className="issue">Chapter 06 · 渲染居中</span>
        </header>
        <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

        <div className="po-intro">
          <div className="kicker">基线公式 · 垂直居中</div>
          <MaskReveal show duration={900}>
            <h2 className="po-head serif-cn">
              文字视觉中心 = <span className="po-em">框中心</span>
            </h2>
          </MaskReveal>
        </div>

        <div className="po-render">
          {/* old formula */}
          <div className="po-formula po-formula-old">
            <div className="po-formula-label label-mono">旧 · 固定 20%</div>
            <code className="po-code">text_y = bbox_bottom + bh * 0.2</code>
            <div className="po-box po-box-old">
              <span className="po-box-text po-box-text-old">随</span>
            </div>
          </div>

          {/* arrow */}
          <svg className="po-arrow" viewBox="0 0 80 24">
            <path d="M 4 12 L 70 12 M 60 4 L 72 12 L 60 20" />
          </svg>

          {/* new formula */}
          <MaskReveal show delay={600} duration={900}>
            <div className="po-formula po-formula-new">
              <div className="po-formula-label label-mono">新 · 垂直居中</div>
              <code className="po-code po-code-new">
                text_y = bbox_bottom + (bh - font_size × 0.8) / 2
              </code>
              <div className="po-box po-box-new">
                <span className="po-box-text po-box-text-new">随</span>
              </div>
            </div>
          </MaskReveal>
        </div>

        <MaskReveal show delay={1500} duration={800}>
          <p className="po-foot serif-cn">
            不管框多大 · 文字都<span className="po-em">正好落在手写痕迹上</span>
          </p>
        </MaskReveal>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="po-scene scene-pad">
        <header className="masthead">
          <span className="brand">默写批改台</span>
          <span className="issue">Chapter 06 · 真实置信度</span>
        </header>
        <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

        <div className="po-intro">
          <div className="kicker">门控阈值 · 三态标注</div>
          <MaskReveal show duration={900}>
            <h2 className="po-head serif-cn">
              模型真实 conf · <span className="po-em">替换伪置信度</span>
            </h2>
          </MaskReveal>
        </div>

        <div className="po-conf">
          <div className="po-legend">
            <div className="po-leg po-leg-green">
              <span className="po-leg-ch">字</span>
              <span className="po-leg-label">green</span>
              <span className="po-leg-val">≥ 0.85</span>
            </div>
            <div className="po-leg po-leg-orange">
              <span className="po-leg-ch">字</span>
              <span className="po-leg-label">orange</span>
              <span className="po-leg-val">≥ 0.60</span>
            </div>
            <div className="po-leg po-leg-skip">
              <span className="po-leg-ch">字</span>
              <span className="po-leg-label">skip</span>
              <span className="po-leg-val">&lt; 0.30</span>
            </div>
          </div>

          <div className="po-conf-compare">
            <div className="po-conf-old">
              <div className="po-conf-label label-mono">旧 · 规则估算</div>
              <div className="po-conf-val po-conf-val-old">0.85</div>
              <div className="po-conf-note">门控形同虚设</div>
            </div>
            <MaskReveal show delay={800} duration={900}>
              <div className="po-conf-new">
                <div className="po-conf-label label-mono">新 · 模型真实</div>
                <div className="po-conf-val po-conf-val-new">0.84</div>
                <div className="po-conf-note">真的低于 0.85 → 标橙</div>
              </div>
            </MaskReveal>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="po-scene scene-pad">
      <header className="masthead">
        <span className="brand">默写批改台</span>
        <span className="issue">Chapter 06 · 7 个存疑字救回</span>
      </header>
      <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

      <div className="po-rescue">
        <div className="kicker">7 个原本漏掉的存疑字</div>
        <MaskReveal show duration={900}>
          <h2 className="po-head serif-cn">
            <span className="po-em">7</span> 个字 · 被正确标橙
          </h2>
        </MaskReveal>

        <div className="po-flip-grid">
          {Array.from({ length: 7 }).map((_, i) => (
            <div
              key={i}
              className="po-flip-cell"
              style={{ animationDelay: `${i * 150 + 400}ms` }}
            >
              <div className="po-flip-inner">
                <div className="po-flip-front">?</div>
                <div className="po-flip-back">?</div>
              </div>
            </div>
          ))}
        </div>

        <MaskReveal show delay={1800} duration={900}>
          <p className="po-foot serif-cn">
            <span className="po-em">绿色 → 橙色</span> · 不再漏判
          </p>
        </MaskReveal>
      </div>
    </div>
  );
}
