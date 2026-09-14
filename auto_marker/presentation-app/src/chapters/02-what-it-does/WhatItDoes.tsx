import { MaskReveal } from "../../components/MaskReveal";
import type { ChapterStepProps } from "../../registry/types";
import "./WhatItDoes.css";

export default function WhatItDoesChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="wd-scene scene-pad">
        <header className="masthead">
          <span className="brand">默写批改台</span>
          <span className="issue">Chapter 02 · 工具亮相</span>
        </header>
        <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

        <div className="wd-pipeline">
          <div className="kicker">5 秒搞定一本</div>
          <div className="wd-flow">
            {/* PDF input */}
            <MaskReveal show duration={700}>
              <div className="wd-card wd-pdf">
                <div className="wd-pdf-page" />
                <div className="wd-pdf-page wd-pdf-page-2" />
                <div className="wd-pdf-label label-mono">PDF 扫描页</div>
              </div>
            </MaskReveal>

            {/* arrow + progress */}
            <div className="wd-bridge">
              <div className="wd-progress-wrap">
                <div className="wd-progress-bar" />
                <span className="wd-progress-text label-mono">5s</span>
              </div>
              <svg className="wd-arrow" viewBox="0 0 80 24">
                <path d="M 4 12 L 70 12 M 60 4 L 72 12 L 60 20" />
              </svg>
            </div>

            {/* output report */}
            <MaskReveal show delay={900} duration={800}>
              <div className="wd-card wd-report">
                <div className="wd-report-line">
                  <span className="wd-ch wd-ch-correct">随</span>
                  <span className="wd-ch wd-ch-correct">君</span>
                  <span className="wd-ch wd-ch-correct">直</span>
                  <span className="wd-ch wd-ch-correct">到</span>
                  <span className="wd-ch wd-ch-correct">夜</span>
                  <span className="wd-ch wd-ch-correct">郎</span>
                  <span className="wd-ch wd-ch-correct">西</span>
                </div>
                <div className="wd-report-line">
                  <span className="wd-ch wd-ch-correct">海</span>
                  <span className="wd-ch wd-ch-correct">内</span>
                  <span className="wd-ch wd-ch-correct">存</span>
                  <span className="wd-ch wd-ch-wrong">知</span>
                  <span className="wd-ch wd-ch-uncertain">己</span>
                </div>
                <div className="wd-report-label label-mono">批改报告</div>
              </div>
            </MaskReveal>
          </div>

          <div className="wd-legend">
            <span className="wd-leg wd-leg-correct">绿勾 · 正确</span>
            <span className="wd-leg wd-leg-wrong">红叉 · 错误</span>
            <span className="wd-leg wd-leg-uncertain">橙问号 · 看不清</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="wd-scene scene-pad">
      <header className="masthead">
        <span className="brand">默写批改台</span>
        <span className="issue">Chapter 02 · 核心引擎</span>
      </header>
      <hr className="rule" style={{ marginTop: "var(--space-5)" }} />

      <div className="wd-engine">
        <div className="kicker">视觉模型 · Qwen3-VL</div>
        <MaskReveal show duration={1000}>
          <h2 className="wd-model-name">
            <span className="serif-it wd-em">Qwen3-VL</span>
          </h2>
        </MaskReveal>
        <MaskReveal show delay={300} duration={900}>
          <p className="wd-model-sub serif-cn">给一张扫描页，认出每个手写汉字 + 坐标</p>
        </MaskReveal>

        <div className="wd-abilities">
          {[
            { n: "01", t: "逐字识别", d: "整篇默写拆成单字，一个一个认", delay: 600 },
            { n: "02", t: "坐标输出", d: "每个字带 bbox，定位到像素", delay: 850 },
            { n: "03", t: "置信度", d: "每个字附带识别可信度分数", delay: 1100 },
          ].map((a) => (
            <MaskReveal key={a.n} show delay={a.delay} duration={800}>
              <div className="wd-ability card">
                <div className="wd-ability-num hero-num">{a.n}</div>
                <div className="wd-ability-body">
                  <div className="wd-ability-title serif-cn">{a.t}</div>
                  <div className="wd-ability-desc label-mono">{a.d}</div>
                </div>
              </div>
            </MaskReveal>
          ))}
        </div>
      </div>
    </div>
  );
}
