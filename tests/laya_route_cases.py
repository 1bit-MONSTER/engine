#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Writes tests/laya_route_cases.json: 200 hand-labelled chat requests, 50 per request class
(code, prose, short, long_doc), for the Laya request classifier (docs/laya.md, RFC #186).
The long_doc bodies are generated here (logs, clauses, tables, listings, notes), so nothing
third-party is copied. Deterministic: rerun to regenerate the same file."""
import json
import random
import sys

CODE = [
    "Write a Python function that parses an ISO-8601 timestamp without using datetime, with tests.",
    "Fix this: my React component re-renders forever when I call setState inside useEffect.",
    "Implement an LRU cache in C++ with O(1) get and put.",
    "Write a bash script that renames every .jpeg file in a folder to .jpg.",
    "How do I read a CSV file into a pandas DataFrame and drop rows with missing values?",
    "Convert this for loop to a list comprehension: result = []; for x in data: if x > 0: result.append(x * 2)",
    "Write a Rust function that reverses a linked list.",
    "Why does this SQL query return duplicates? SELECT u.name FROM users u JOIN orders o ON o.user_id = u.id",
    "Write a Go HTTP handler that returns the current time as JSON.",
    "Refactor this JavaScript to use async/await instead of .then chains: fetch(url).then(r => r.json()).then(d => render(d))",
    "Write a unit test for a function that checks whether a string is a palindrome.",
    "My Dockerfile builds but the container exits immediately. Here is the CMD line: CMD python app.py &",
    "Write a regex that matches IPv4 addresses.",
    "Implement binary search in Java and explain the off-by-one pitfalls in comments.",
    "Write a CMake file for a library with two source files and a test executable.",
    "How do I debounce a search input in TypeScript?",
    "Write a GitHub Actions workflow that runs pytest on push.",
    "Segfault in my C code when freeing a struct array, what am I doing wrong? free(arr); free(arr[0].name);",
    "Write a Kotlin data class for a user with validation of the email field.",
    "Write a SQL migration that adds a nullable column and backfills it from another table.",
    "Explain and fix: TypeError: 'NoneType' object is not subscriptable on line data['items'][0]",
    "Write a Python decorator that retries a function three times with exponential backoff.",
    "Implement a thread-safe queue in C++17 with a condition variable.",
    "Write a small Flask app with one POST endpoint that stores JSON in SQLite.",
    "Port this Python to C: def mean(xs): return sum(xs) / len(xs)",
    "Write a Makefile rule that rebuilds objects when headers change.",
    "How do I profile a slow Node.js function?",
    "Write a HIP kernel that adds two float arrays.",
    "Write a shell one-liner to find the ten largest files under /var.",
    "Write a Swift function that downloads an image and caches it on disk.",
    "Make this Python faster: for i in range(len(a)): for j in range(len(b)): if a[i] == b[j]: count += 1",
    "Write a Terraform resource for an S3 bucket with versioning enabled.",
    "Write a PyTorch training loop with gradient clipping and a learning-rate schedule.",
    "How do I mock an HTTP call in a Jest test?",
    "Write a C function that parses a hex string into bytes and rejects bad input.",
    "Implement Dijkstra's algorithm in Python with a heap.",
    "Write a git pre-commit hook that runs clang-format on staged files.",
    "My Vulkan compute shader gives wrong results when the workgroup size is 64 but not 32, why?",
    "Write a Haskell function that counts word frequencies in a string.",
    "Write an nginx config that proxies /api to localhost:8080 with websockets.",
    "Write a llama.cpp command line that serves a GGUF model on port 8080 with flash attention.",
    "Write a Python script that watches a directory and prints new files.",
    "Implement a trie in TypeScript with insert and prefix search.",
    "Convert this callback-based Node code to use promises: fs.readFile(p, (err, d) => cb(err, d))",
    "Write a SQL query for the top three products by revenue per month.",
    "Write a Zig program that prints the first 20 Fibonacci numbers.",
    "Why is my Python multiprocessing Pool hanging on macOS?",
    "Write an Arduino sketch that blinks an LED faster when a button is held.",
    "Write a Dockerfile for a Rust binary using a multi-stage build.",
    "Add type hints to this function: def merge(a, b, key): return {**a, **b, key: a.get(key) or b.get(key)}",
]
PROSE = [
    "Explain how a B-tree insert works, step by step.",
    "What are the main causes of inflation, and how do central banks respond?",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Compare the leadership styles of Lincoln and Churchill.",
    "Explain quantum entanglement to a curious teenager.",
    "Write a cover letter for a junior data analyst position.",
    "What should I consider when choosing between renting and buying a home?",
    "Describe the water cycle and why it matters for agriculture.",
    "Write a blog post about the benefits of learning a second language as an adult.",
    "Explain the difference between weather and climate, with examples.",
    "How did the printing press change European society?",
    "Give me a week-long vegetarian meal plan with a shopping list.",
    "Write a persuasive essay arguing for four-day work weeks.",
    "Explain how vaccines train the immune system.",
    "What are the pros and cons of electric cars today?",
    "Write a eulogy for a beloved grandmother who loved gardening.",
    "Explain the plot and themes of Moby-Dick.",
    "How do I prepare for a job interview at a startup?",
    "Describe how the internet routes a packet from my laptop to a website.",
    "Write a product description for a handmade ceramic coffee mug.",
    "Explain the causes of the First World War.",
    "What is stoicism and how can I apply it to daily life?",
    "Write a speech for a friend's wedding, warm and a little funny.",
    "Explain how neural networks learn, without math.",
    "Plan a three-day trip to Lisbon for someone who loves food and history.",
    "Why is sleep important, and how can I sleep better?",
    "Write a letter to my landlord asking for a repair, polite but firm.",
    "Explain supply and demand using the example of concert tickets.",
    "Describe the life cycle of a star.",
    "Write a short poem about autumn in the city and explain the imagery.",
    "How should a small team run effective weekly meetings?",
    "Explain what a mortgage is and how interest is calculated over time.",
    "What happened during the Apollo 13 mission?",
    "Write a motivational message for a team that just missed a deadline.",
    "Explain the difference between a virus and a bacterium.",
    "How do I start running if I've never exercised?",
    "Write the opening chapter of a mystery set on a night train.",
    "What makes a good user interface, in your view?",
    "Explain photosynthesis in simple terms.",
    "How do elections work in a parliamentary system?",
    "Write a review of an imaginary Italian restaurant.",
    "Explain why the sky is blue and sunsets are red.",
    "Give advice to someone moving to a new city alone.",
    "Explain the history and purpose of the Olympic Games.",
    "Write a dialogue between a skeptic and a believer about astrology.",
    "What is the role of the mitochondria in a cell?",
    "Describe the main schools of thought in ethics.",
    "Write a newsletter intro announcing a community garden opening.",
    "How does compound interest work, and why does starting early matter?",
    "Explain how Strix Halo differs from a desktop CPU with a discrete GPU.",
]
SHORT = [
    "hi", "What is 17*23?", "Translate to French: the quick brown fox.", "Capital of Australia?",
    "thanks!", "Is 91 prime?", "What year did the Berlin Wall fall?", "Spell 'necessary'.",
    "Convert 72 Fahrenheit to Celsius.", "What's the plural of cactus?", "Synonym for happy?",
    "How many ounces in a pound?", "Translate 'good morning' to Japanese.", "Who wrote Hamlet?",
    "What does CPU stand for?", "Round 3.14159 to two decimals.", "ok", "Define 'ephemeral'.",
    "How many days in a leap year?", "Is a tomato a fruit?", "What's 15% of 80?",
    "Largest planet in the solar system?", "yes please", "What time zone is Halifax in?",
    "Translate to Spanish: where is the train station?", "Boiling point of water in Kelvin?",
    "What's the square root of 144?", "Who painted the Mona Lisa?", "Opposite of 'generous'?",
    "How many continents are there?", "What is H2O?", "hello, are you there?",
    "Chemical symbol for gold?", "What's 2 to the power of 10?", "Is Pluto a planet?",
    "What language is spoken in Brazil?", "How many sides does a hexagon have?",
    "Translate 'thank you' to German.", "Who was the first person on the Moon?",
    "What does GPU stand for?", "cool, thanks", "What's the speed of light in km/s?",
    "Abbreviation for kilobyte?", "Which is bigger, 0.7 or 0.65?", "Name a primary color.",
    "What is the freezing point of water in Fahrenheit?", "How many minutes in three hours?",
    "Rhymes with 'cat'?", "Is it 'affect' or 'effect' for the result of something?", "good night",
]
LONG_INSTR = [
    "Summarize this document and list every action item:",
    "Find the errors in these logs and tell me the root cause:",
    "Review this contract and list every termination clause:",
    "Here is our full meeting transcript. Write the minutes:",
    "Analyze this table and describe the trends:",
    "Review this whole file for bugs:",
    "Translate this entire document to French:",
    "Compare these two policy versions and list every change:",
    "Extract all names, dates and amounts from this report:",
    "Answer questions about this manual. First, what does section 3 say?",
]


def body(kind, r):
    if kind == 0:
        return "\n".join(f"- {r.choice(['Ana','Ben','Chen','Dara','Eli'])} to {r.choice(['update the budget','ship the beta','review the vendor quote','draft the FAQ','fix the login bug','schedule the audit'])} by {r.choice(['Monday','Friday','the 12th','end of month'])}. Notes: {' '.join(r.choice(['discussed','agreed','blocked','pending','approved','risk','owner','scope']) for _ in range(12))}." for _ in range(30))
    if kind == 1:
        return "\n".join(f"2026-09-{r.randint(1,27):02d}T{r.randint(0,23):02d}:{r.randint(0,59):02d}:{r.randint(0,59):02d}Z {r.choice(['INFO','WARN','ERROR','DEBUG'])} {r.choice(['api','db','worker','auth','cache'])}[{r.randint(100,999)}]: {r.choice(['request served','timeout after 30s','connection reset by peer','retrying','pool exhausted','cache miss','token expired','slow query 2.3s'])} id={r.randint(10000,99999)}" for _ in range(45))
    if kind == 2:
        return "\n\n".join(f"{i}.{j} {r.choice(['Either party','The Supplier','The Customer','The Licensor'])} may {r.choice(['terminate this Agreement','suspend the Services','assign its rights','renew the Term'])} {r.choice(['upon thirty (30) days written notice','immediately upon material breach','if payment is overdue by sixty (60) days','at the end of the Initial Term'])}, subject to Section {r.randint(1,14)}.{r.randint(1,9)} and the limitations in Schedule {r.choice('ABCD')}." for i in range(1, 9) for j in range(1, 4))
    if kind == 3:
        return "\n".join(f"{r.choice(['ANA','BEN','CHEN','DARA'])}: {' '.join(r.choice(['I think','we should','the numbers','next quarter','customers','the release','honestly','agreed','but','risk','timeline','budget','hiring']) for _ in range(r.randint(8,20)))}." for _ in range(35))
    if kind == 4:
        return "month,region,units,revenue,returns\n" + "\n".join(f"2026-{m:02d},{reg},{r.randint(100,999)},{r.randint(1000,99999)},{r.randint(0,40)}" for m in range(1, 13) for reg in ('north', 'south', 'east', 'west'))
    if kind == 5:
        return "\n".join(f"{'    ' * r.randint(0,2)}{r.choice(['int','if (','for (int i = 0; i < n; i++)','return','auto','std::vector<float>','while (','x +=','free(','memcpy('])} {r.choice(['buf','len','ptr','count','node->next','out[i]','cfg.size'])}{r.choice([';',') {','];',' * 2;',', n);'])}" for _ in range(60))
    words = ['the','system','shall','provide','users','with','access','to','reports','each','quarter','and','store','records','for','seven','years','under','policy','review']
    return "\n\n".join(f"Section {i}. " + " ".join(r.choice(words) for _ in range(70)) + "." for i in range(1, 8))


def main():
    r = random.Random(186)
    cases = [{"class": "code", "text": t} for t in CODE] + [{"class": "prose", "text": t} for t in PROSE] \
        + [{"class": "short", "text": t} for t in SHORT]
    for i in range(50):
        k = i % 7 if i % 10 != 6 else 6
        instr = LONG_INSTR[i % len(LONG_INSTR)]
        cases.append({"class": "long_doc", "text": f"{instr}\n\n{body(k, r)}"})
    assert all(sum(c["class"] == k for c in cases) == 50 for k in ("code", "prose", "short", "long_doc"))
    out = sys.argv[1] if len(sys.argv) > 1 else "tests/laya_route_cases.json"
    json.dump({"about": "RFC #186: labelled requests for the Laya request classifier; regenerate with tests/laya_route_cases.py",
               "classes": ["code", "prose", "short", "long_doc"], "cases": cases}, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"{len(cases)} cases -> {out}")


if __name__ == "__main__":
    main()
