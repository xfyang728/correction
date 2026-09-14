import { MaskReveal } from "../../components/MaskReveal";
import type { ChapterStepProps } from "../../registry/types";
import "./YFix.css";

export default function YFixChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="yf-scene scene-pad">
        <header className="masthead">
          <span className="brand">默写批改台</span>
          <span className="issue">Chapter 04 · Y-Clamp</span>
        </header>
        <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

        <div className="yf-intro">
          <div className="kicker">修正策略 · _clamp_chars_y_to_region()</div>
          <MaskReveal show duration={900}>
            <h2 className="yf-name serif-cn">
              <span className="yf-em">Y-Clamp</span>
            </h2>
          </MaskReveal>
          <MaskReveal show delay={400} duration={800}>
            <p className="yf-tagline serif-cn">把模型 Y 替换成手写框真实 Y 范围</p>
          </MaskReveal>
        </div>

        <div className="yf-diagram">
          {/* Before */}
          <div className="yf-side yf-before">
            <div className="yf-side-label label-mono">Before · 模型输出</div>
            <div className="yf-frame">
              <div className="yf-box yf-box-wrong" style={{ top: "30%" }}>题1</div>
              <div className="yf-box yf-box-wrong" style={{ top: "32%" }}>题5</div>
              <div className="yf-box yf-box-wrong" style={{ top: "34%" }}>题7</div>
            </div>
          </div>

          {/* Arrow */}
          <div className="yf-bridge">
            <div className="yf-fn">_clamp_chars_y_to_region()</div>
            <svg className="yf-arrow" viewBox="0 0 80 24">
              <path d="M 4 12 L 70 12 M 60 4 L 72 12 L 60 20" />
            </svg>
            <div className="yf-snap">啪！</div>
          </div>

          {/* After */}
          <div className="yf-side yf-after">
            <div className="yf-side-label label-mono">After · Clamp 修正</div>
            <div className="yf-frame">
              <MaskReveal show delay={800} duration={900}>
                <div className="yf-box yf-box-right" style={{ top: "15%" }}>题1</div>
              </MaskReveal>
              <MaskReveal show delay={1100} duration={900}>
                <div className="yf-box yf-box-right" style={{ top: "45%" }}>题5</div>
              </MaskReveal>
              <MaskReveal show delay={1400} duration={900}>
                <div className="yf-box yf-box-right" style={{ top: "75%" }}>题7</div>
              </MaskReveal>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="yf-scene scene-pad">
      <header className="masthead">
        <span className="brand">默写批改台</span>
        <span className="issue">Chapter 04 · 保留相对差异</span>
      </header>
      <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

      <div className="yf-detail">
        <div className="kicker">不一刀切 · 按比例保留</div>
        <MaskReveal show duration={900}>
          <h2 className="yf-detail-head serif-cn">
            逐字高低差异 <span className="yf-em">保留</span>
          </h2>
        </MaskReveal>

        <div className="yf-zoom">
          <div className="yf-zoom-label label-mono">放大看两个字</div>
          <div className="yf-zoom-frame">
            {/* handwriting region */}
            <div className="yf-zoom-region">
              <div className="yf-zoom-line" />
            </div>
            {/* model Y — has tiny relative diff */}
            <div className="yf-zoom-model">
              <div className="yf-zoom-ch yf-zoom-ch-a" style={{ top: "20%" }}>
                <span className="yf-zoom-char">随</span>
                <span className="yf-zoom-val">y=0.46</span>
              </div>
              <div className="yf-zoom-ch yf-zoom-ch-b" style={{ top: "26%" }}>
                <span className="yf-zoom-char">君</span>
                <span className="yf-zoom-val">y=0.48</span>
              </div>
            </div>
            {/* arrow */}
            <div className="yf-zoom-arrow">→</div>
            {/* after clamp — preserves diff */}
            <div className="yf-zoom-clamped">
              <div className="yf-zoom-ch yf-zoom-ch-a" style={{ top: "30%" }}>
                <span className="yf-zoom-char">随</span>
                <span className="yf-zoom-val">y=0.20</span>
              </div>
              <div className="yf-zoom-ch yf-zoom-ch-b" style={{ top: "36%" }}>
                <span className="yf-zoom-char">君</span>
                <span className="yf-zoom-val">y=0.22</span>
              </div>
            </div>
          </div>
          <MaskReveal show delay={1200} duration={800}>
            <p className="yf-zoom-foot serif-cn">
              模型给的<span className="yf-em">相对高低</span> · 映射后<span className="yf-em">依然保留</span>
            </p>
          </MaskReveal>
        </div>
      </div>
    </div>
  );
}
