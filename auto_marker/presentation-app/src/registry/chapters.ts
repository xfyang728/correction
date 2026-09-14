import type { ChapterDef } from "./types";
import ColdopenChapter from "../chapters/01-coldopen/Coldopen";
import { narrations as coldopenNarrations } from "../chapters/01-coldopen/narrations";
import WhatItDoesChapter from "../chapters/02-what-it-does/WhatItDoes";
import { narrations as whatItDoesNarrations } from "../chapters/02-what-it-does/narrations";
import TheProblemChapter from "../chapters/03-the-problem/TheProblem";
import { narrations as theProblemNarrations } from "../chapters/03-the-problem/narrations";
import YFixChapter from "../chapters/04-y-fix/YFix";
import { narrations as yFixNarrations } from "../chapters/04-y-fix/narrations";
import XFixChapter from "../chapters/05-x-fix/XFix";
import { narrations as xFixNarrations } from "../chapters/05-x-fix/narrations";
import PolishChapter from "../chapters/06-polish/Polish";
import { narrations as polishNarrations } from "../chapters/06-polish/narrations";
import ResultsChapter from "../chapters/07-results/Results";
import { narrations as resultsNarrations } from "../chapters/07-results/narrations";

export const CHAPTERS: ChapterDef[] = [
  {
    id: "coldopen",
    title: "痛点钩子",
    narrations: coldopenNarrations,
    Component: ColdopenChapter,
  },
  {
    id: "what-it-does",
    title: "工具亮相",
    narrations: whatItDoesNarrations,
    Component: WhatItDoesChapter,
  },
  {
    id: "the-problem",
    title: "坐标全偏了",
    narrations: theProblemNarrations,
    Component: TheProblemChapter,
  },
  {
    id: "y-fix",
    title: "Y-Clamp 修正",
    narrations: yFixNarrations,
    Component: YFixChapter,
  },
  {
    id: "x-fix",
    title: "首字偏右修复",
    narrations: xFixNarrations,
    Component: XFixChapter,
  },
  {
    id: "polish",
    title: "渲染居中 + 真实置信度",
    narrations: polishNarrations,
    Component: PolishChapter,
  },
  {
    id: "results",
    title: "实测结果 + CTA",
    narrations: resultsNarrations,
    Component: ResultsChapter,
  },
];
