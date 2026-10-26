import asyncio
import json
import os
import shutil
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM_PATH") or shutil.which("chromium"),
            headless=True,
            args=["--no-sandbox"],
        )
        page = await b.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.goto(os.environ.get("MATCHLAB_URL", "http://127.0.0.1:8001/"))
        await page.locator("#champion").filter(has_text="FC Barcelona").wait_for()
        await page.click("#submit")
        await page.wait_for_function("document.querySelector('#p-home').textContent.includes('%')")
        await page.screenshot(path=str(ROOT / "docs/desktop.png"), full_page=True)
        await page.select_option("#competition", "CL")
        await page.wait_for_function(
            "document.querySelector('#champion').textContent.includes('Paris')"
        )
        await page.select_option("#home", "131")
        await page.select_option("#away", "27")
        await page.click("#submit")
        await page.wait_for_function(
            "document.querySelector('#match-title').textContent.includes('Bayern')"
        )
        assert await page.locator("#error").inner_text() == ""
        await page.screenshot(path=str(ROOT / "docs/europe.png"), full_page=True)
        for c, champ in [
            ("GB1", "Liverpool"),
            ("FR1", "Paris"),
            ("L1", "Bayern"),
            ("IT1", "Napoli"),
            ("EL", "Tottenham"),
            ("UCOL", "Chelsea"),
        ]:
            await page.select_option("#competition", c)
            await page.wait_for_function(
                '(name)=>document.querySelector("#champion").textContent.includes(name)', arg=champ
            )
            await page.wait_for_timeout(150)
        await page.select_option("#competition", "ES1")
        await page.select_option("#season", "2018")
        await page.wait_for_function(
            "document.querySelector('#value-date').textContent.includes('2019-06-30')"
        )
        assert "Lionel Messi" in await page.locator("#value-rows").inner_text()
        await page.set_viewport_size({"width": 390, "height": 844})
        await page.screenshot(path=str(ROOT / "docs/mobile.png"), full_page=True)
        overflow = await page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
        assert not overflow, "Page overflows mobile viewport"
        assert not errors, errors
        (ROOT / "docs/browser-results.json").write_text(
            json.dumps(
                {
                    "desktop": [1440, 1100],
                    "mobile": [390, 844],
                    "javascript_errors": errors,
                    "mobile_page_overflow": overflow,
                    "checks": [
                        "ES1 real prediction",
                        "CL Barcelona vs Bayern cross-league prediction",
                        "All eight competition selectors",
                        "2018/19 Barcelona valuation includes Messi",
                        "No page-wide horizontal overflow",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        await b.close()


asyncio.run(main())
