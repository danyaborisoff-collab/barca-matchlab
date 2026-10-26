import {readFileSync} from "node:fs";
import assert from "node:assert/strict";
const html=readFileSync("web/index.html","utf8");
const js=readFileSync("web/app.js","utf8");
for(const match of js.matchAll(/\$\("([\w-]+)"\)/g)) assert(html.includes(`id="${match[1]}"`),`Missing DOM id: ${match[1]}`);
assert(!js.includes("innerHTML"),"Unsafe DOM insertion");
assert(html.includes('lang="ru"'));
console.log("Frontend contract checks passed");
