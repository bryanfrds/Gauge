# YN — Plain-English Overview

## What it is

YN is a small AI that **makes choices**. You give it:

1. some **input**, like a message, a product description or a support ticket,
2. a **question** about that input, and
3. a **list of possible answers**.

It picks one answer and gives a **confidence score** from 0 to 1, where 1 means certain.

The simplest case is yes or no, which is where the name comes from. The same model also
handles longer lists, like picking one of five teams.

## What it is not

- **Not a chatbot.** It never writes sentences, explanations or summaries.
- **Not a know-it-all.** It judges the input you give it. It doesn't look things up
  or know recent news.
- **Not a free-for-all.** It can only pick from the answers you list, so it can't
  invent a surprise answer.

## Why that's useful

| | Chat AI (like ChatGPT) | YN |
|---|---|---|
| Output | Paragraphs of text | One answer + a confidence number |
| Speed | Seconds | Milliseconds (thousandths of a second) |
| Cost per decision | Cents | A tiny fraction of a cent |
| Can the answer be anything? | Yes, so it can go off-script | No, only the answers you allowed |
| Runs on your own computer? | Usually needs a big graphics card | Yes, on a normal processor |

## When to trust the confidence

The confidence number is the most important part. The goal:

- **Above ~0.9:** act on it automatically.
- **Between ~0.6 and 0.9:** act on it, but spot-check.
- **Below ~0.6:** send it to a person.

These cutoffs are starting points. Each app should set its own after testing on its
real data.

## Example uses

- Sorting incoming emails or support tickets into the right queue
- Flagging spam or unsafe content
- Scoring sales leads as hot / warm / cold
- Checking whether a document mentions a required item (yes / no)
- Routing a request to the right tool or AI agent

## What "open source" means here

Everything is public and free to use, including in commercial products:

- the code to run it,
- the trained model itself,
- how it was trained and on what data,
- how it was tested and the results.

Anyone can check the test results for themselves.
