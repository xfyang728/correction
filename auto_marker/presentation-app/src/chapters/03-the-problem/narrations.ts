import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "听起来很美好对吧。但实际跑起来你会发现一个很尴尬的事。模型认字没问题，坐标全偏了。",
  "我拿到第一版结果的时候，整个批注报告的文字位置整体偏下，每行第一个字还整体偏右。看着就像把字幕放错了位置。",
  "我查了一圈，发现这是 Qwen3-VL 的一个老毛病。它的 Y 坐标有系统性偏移，经常好几道题共用同一个 Y 值。",
];
