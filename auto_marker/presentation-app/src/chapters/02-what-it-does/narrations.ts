import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "所以我自己做了一个工具。丢一张 PDF 扫描页进去，5 秒出来一份批改完的报告。正确打绿勾，错误打红叉，看不清的标橙色问号。",
  "它的核心是一个叫 Qwen3-VL 的视觉模型。你给它一张学生默写的扫描页，它能逐个认出学生手写的汉字，还带上每个字在页面上的坐标。",
];
