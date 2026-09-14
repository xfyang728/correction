import { MaskReveal } from "../../components/MaskReveal";
import type { ChapterStepProps } from "../../registry/types";
import "./Results.css";

export default function ResultsChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="rs-scene scene-pad">
        <header className="masthead">
          <span className="brand">默写批改台</span>
          <span className="issue">Chapter 07 · 实测结果</span>
        </header>
        <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

        <div className="rs-intro">
          <div className="kicker">真实样本 · 跑完所有修复</div>
          <MaskReveal show duration={900}>
            <h2 className="rs-head serif-cn">数字说话</h2>
          </MaskReveal>
        </div>

        <div className="rs-stats">
          <MaskReveal show delay={300} duration={800}>
            <div className="rs-stat">
              <div className="rs-stat-num hero-num">62</div>
              <div className="rs-stat-label label-mono">总字数</div>
            </div>
          </MaskReveal>
          <MaskReveal show delay={700} duration={800}>
            <div className="rs-stat rs-stat-green">
              <div className="rs-stat-num hero-num">61</div>
              <div className="rs-stat-label label-mono">正确识别</div>
            </div>
          </MaskReveal>
          <MaskReveal show delay={1100} duration={800}>
            <div className="rs-stat rs-stat-orange">
              <div className="rs-stat-num hero-num">1</div>
              <div className="rs-stat-label label-mono">标存疑</div>
            </div>
          </MaskReveal>
          <MaskReveal show delay={1500} duration={900}>
            <div className="rs-stat rs-stat-zero">
              <div className="rs-stat-num hero-num">0</div>
              <div className="rs-stat-label label-mono">错</div>
            </div>
          </MaskReveal>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="rs-scene scene-pad">
        <header className="masthead">
          <span className="brand">默写批改台</span>
          <span className="issue">Chapter 07 · 坐标精度</span>
        </header>
        <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

        <div className="rs-intro">
          <div className="kicker">坐标误差 · 压缩 57%</div>
          <MaskReveal show duration={900}>
            <h2 className="rs-head serif-cn">
              144px → <span className="rs-em">61px</span>
            </h2>
          </MaskReveal>
        </div>

        <div className="rs-chart">
          <div className="rs-chart-axis">
            <span className="rs-chart-tick">150</span>
            <span className="rs-chart-tick">100</span>
            <span className="rs-chart-tick">50</span>
            <span className="rs-chart-tick">0</span>
          </div>
          <div className="rs-chart-bars">
            <div className="rs-bar-wrap">
              <div className="rs-bar rs-bar-old">
                <span className="rs-bar-val">144px</span>
              </div>
              <span className="rs-bar-label label-mono">修复前</span>
            </div>
            <div className="rs-bar-wrap">
              <div className="rs-bar rs-bar-new">
                <span className="rs-bar-val">61px</span>
              </div>
              <span className="rs-bar-label label-mono">修复后</span>
            </div>
          </div>
          <div className="rs-improve">
            <span className="rs-improve-num">-57%</span>
            <span className="rs-improve-label label-mono">误差压缩</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rs-scene scene-pad rs-final-scene">
      <header className="masthead">
        <span className="brand">默写批改台</span>
        <span className="issue">Chapter 07 · 开源 · 评论区置顶</span>
      </header>
      <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

      <div className="rs-final">
        <div className="rs-final-report">
          <div className="rs-final-page">
            <div className="rs-final-line">
              <span className="rs-final-marker">1.</span>
              {Array.from("随君直到夜郎西").map((c, i) => (
                <span key={i} className="rs-final-ch rs-final-ch-correct">{c}</span>
              ))}
            </div>
            <div className="rs-final-line">
              <span className="rs-final-marker">2.</span>
              {Array.from("海内存知己").map((c, i) => (
                <span key={i} className="rs-final-ch rs-final-ch-correct">{c}</span>
              ))}
              <span className="rs-final-ch rs-final-ch-uncertain rs-final-blink">?</span>
            </div>
            <div className="rs-final-line">
              <span className="rs-final-marker">3.</span>
              {Array.from("天涯若比邻").map((c, i) => (
                <span key={i} className="rs-final-ch rs-final-ch-correct">{c}</span>
              ))}
            </div>
            <div className="rs-final-line">
              <span className="rs-final-marker">4.</span>
              {Array.from("落霞与孤鹜齐飞").map((c, i) => (
                <span key={i} className="rs-final-ch rs-final-ch-correct">{c}</span>
              ))}
            </div>
          </div>
        </div>

        <MaskReveal show delay={800} duration={900}>
          <div className="rs-cta">
            <div className="rs-cta-line serif-cn">
              老师只需扫一眼<span className="rs-em">橙色问号</span>
            </div>
            <div className="rs-cta-sub serif-cn">
              工具已开源 · 链接在<span className="rs-em">评论区置顶</span>
            </div>
            <div className="rs-cta-meta label-mono">
              <span className="dot-accent" />&nbsp;Qwen3-VL · Python · 开源
            </div>
          </div>
        </MaskReveal>
      </div>
    </div>
  );
}
