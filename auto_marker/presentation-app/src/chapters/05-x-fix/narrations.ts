import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "修完 Y 我发现 X 也偏。每行第一个字比实际位置偏右 90 到 130 像素。原因出在一个等宽假设上。我在算答案起始位置时，假设每个字符宽度一样。但题号是 ASCII 的括号加数字，答案才是中文。中文字符实际宽度是 ASCII 的两倍。",
  "修法是用 unicodedata 的 east_asian_width 函数。中文字符权重给 2，ASCII 给 1，按真实宽度比例算位置。改完之后首字位置左移了快 100 像素，正好对齐答案起始处。",
];
