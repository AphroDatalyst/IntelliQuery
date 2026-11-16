import { describe, expect, it } from "vitest";

import { formatBytes, highlightStyle, splitAnswer } from "./citations";

describe("splitAnswer", () => {
  it("turns known markers into citations and keeps the text around them", () => {
    expect(splitAnswer("The limit is 8 C. [1] Alarms use SMS. [2][1]", new Set([1, 2]))).toEqual([
      { kind: "text", text: "The limit is 8 C." },
      { kind: "cite", marker: 1 },
      { kind: "text", text: " Alarms use SMS." },
      { kind: "cite", marker: 2 },
      { kind: "cite", marker: 1 },
    ]);
  });

  it("leaves bracketed numbers that are not citations as text", () => {
    expect(splitAnswer("See step [7] of the list. [1]", new Set([1]))).toEqual([
      { kind: "text", text: "See step [7] of the list." },
      { kind: "cite", marker: 1 },
    ]);
  });

  it("returns the whole text when there is nothing to cite", () => {
    expect(splitAnswer("Not in the documents.", new Set())).toEqual([
      { kind: "text", text: "Not in the documents." },
    ]);
    expect(splitAnswer("", new Set([1]))).toEqual([]);
  });
});

describe("highlightStyle", () => {
  const region = { page: 4, page_width: 800, page_height: 1000 };

  it("expresses the box as percentages of the page", () => {
    const box = { x0: 80, y0: 100, x1: 400, y1: 300 };
    expect(highlightStyle({ ...region, box })).toEqual({
      left: "10%",
      top: "10%",
      width: "40%",
      height: "20%",
    });
  });

  it("clamps a box that runs past the page edge", () => {
    const box = { x0: 400, y0: 900, x1: 1200, y1: 1100 };
    expect(highlightStyle({ ...region, box })).toEqual({
      left: "50%",
      top: "90%",
      width: "50%",
      height: "10%",
    });
  });

  it("gives nothing to draw without a box, a page size, or any area", () => {
    expect(highlightStyle({ ...region, box: null })).toBeNull();
    const box = { x0: 1, y0: 1, x1: 5, y1: 5 };
    expect(highlightStyle({ page: 4, page_width: null, page_height: null, box })).toBeNull();
    expect(highlightStyle({ ...region, box: { x0: 5, y0: 5, x1: 5, y1: 9 } })).toBeNull();
  });
});

describe("formatBytes", () => {
  it("picks a readable unit", () => {
    expect([formatBytes(512), formatBytes(20_480), formatBytes(3_670_016)]).toEqual([
      "512 B",
      "20 KB",
      "3.5 MB",
    ]);
  });
});
