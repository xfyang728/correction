import { MaskReveal } from "../../components/MaskReveal";
import type { ChapterStepProps } from "../../registry/types";
import "./Coldopen.css";

export default function ColdopenChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="co-scene scene-pad">
        <header className="masthead">
          <span className="brand">默写批改台</span>
          <span className="issue">Chapter 01 · 痛点</span>
        </header>
        <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

        <div className="co-cover">
          <div className="kicker">一位语文老师的日常</div>
          <div className="co-hero-wrap">
            <MaskReveal show duration={950}>
              <span className="co-hero-num hero-num">40</span>
            </MaskReveal>
            <MaskReveal show delay={350} duration={900}>
              <span className="co-hero-unit">本</span>
            </MaskReveal>
          </div>
          <MaskReveal show delay={700} duration={900}>
            <p className="co-question serif-cn">每天改默写要多久？</p>
          </MaskReveal>
          <div className="co-cue label-mono">
            <span className="dot-accent" />&nbsp;点击翻页
          </div>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="co-scene scene-pad">
        <header className="masthead">
          <span className="brand">默写批改台</span>
          <span className="issue">Chapter 01 · 痛点</span>
        </header>
        <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

        <div className="co-answer">
          <div className="kicker">保守估计</div>
          <div className="co-answer-line">
            <MaskReveal show duration={1000}>
              <span className="co-answer-num hero-num">1</span>
            </MaskReveal>
            <MaskReveal show delay={400} duration={900}>
              <span className="co-answer-unit serif-cn">小时起步</span>
            </MaskReveal>
          </div>
          <svg className="co-strike" viewBox="0 0 800 60" preserveAspectRatio="none">
            <path
              className="co-strike-path"
              d="M 10 30 Q 200 10 400 30 T 790 30"
            />
          </svg>
          <MaskReveal show delay={900} duration={900}>
            <p className="co-answer-foot serif-cn">这还只是<span className="co-em">顺利</span>的情况</p>
          </MaskReveal>
        </div>
      </div>
    );
  }

  return (
    <div className="co-scene scene-pad co-repeat-scene">
      <header className="masthead">
        <span className="brand">默写批改台</span>
        <span className="issue">Chapter 01 · 重复劳动</span>
      </header>
      <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

      <div className="co-repeat">
        <div className="kicker">一遍 · 又一遍 · 又一遍</div>
        <div className="co-chalkboard">
          {[
            { t: "随君直到夜郎西", d: 0 },
            { t: "海内存知己", d: 250 },
            { t: "随君直到夜郎西", d: 500 },
            { t: "海内存知己", d: 750 },
            { t: "随君直到夜郎西", d: 1000 },
            { t: "海内存知己", d: 1250 },
          ].map((line, i) => (
            <MaskReveal key={i} show delay={line.d} duration={700}>
              <div className="co-chalk-line" style={{ opacity: 1 - i * 0.12 }}>
                {line.t}
              </div>
            </MaskReveal>
          ))}
        </div>
        <MaskReveal show delay={1700} duration={900}>
          <p className="co-repeat-foot serif-cn">
            大部分都是<span className="co-em">重复劳动</span>
          </p>
        </MaskReveal>
      </div>
    </div>
  );
}
