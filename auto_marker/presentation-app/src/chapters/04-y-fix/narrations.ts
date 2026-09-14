import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "我的办法是 Y-Clamp。每次识别完，把模型给的 Y 坐标直接替换成手写框的真实 Y 范围。但这里有个细节，如果模型在不同字之间有相对高低差异，我会按比例保留，而不是一刀切。",
  "如果模型在不同字之间有相对高低差异，我会按比例保留，而不是一刀切。",
];
