import { test, expect } from "bun:test";
import { looksLikeReasoning, stripReasoning } from "./announce";

/** The text that actually reached Threads. */
const PUBLISHED_BY_MISTAKE =
  'We need to produce a short post under 400 characters. Must open with concrete surprising ' +
  'thing, not "New post:". Mention interactive simulation. No URL. No hashtags spam, at most ' +
  'one emoji if it earns its place. Should be plain, concrete, no hype. Provide the ' +
  'announcement text only.';

test("catches the exact text that was published by mistake", () => {
  expect(looksLikeReasoning(PUBLISHED_BY_MISTAKE)).not.toBeNull();
});

test("catches other shapes of thinking out loud", () => {
  for (const s of [
    "Let's write something punchy about the paper.",
    "I should mention the simulation and keep it under 400 characters.",
    "Okay, the user wants a post about EchoWM.",
    "First, the announcement text must not include a URL.",
  ]) {
    expect(looksLikeReasoning(s)).not.toBeNull();
  }
});

test("passes a real post", () => {
  const good =
    "Three training sources, one command sequence, three completely different routes — until " +
    "you calibrate them to a shared metric scale. EchoWM drives 720p video, sound, music and " +
    "speech from a single 6-DoF camera intent. There's an interactive map on the page you can " +
    "steer yourself.";
  expect(looksLikeReasoning(good)).toBeNull();
});

test("passes a post that merely starts with a number or a name", () => {
  expect(looksLikeReasoning("720p video, sound and speech from one trajectory signal.")).toBeNull();
  expect(looksLikeReasoning("EchoWM generates four modalities from a single camera intent.")).toBeNull();
});

test("strips explicit reasoning blocks", () => {
  expect(stripReasoning("<think>hmm, what angle</think>\nThe real post.")).toBe("The real post.");
  expect(stripReasoning("<reasoning>plan</reasoning> Actual copy here.")).toBe("Actual copy here.");
});
