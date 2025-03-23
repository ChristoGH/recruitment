#!/usr/bin/env python3
"""
Debug Test Crawler

A simplified script to debug the web crawler issue where:
- The same URL returns success=False with crawl_website_sync
- But works with async_main
"""

import asyncio
import sys
from web_crawler_lib import crawl_website_sync, WebCrawlerResult


async def async_crawler_test(url):
    """Test the AsyncWebCrawler directly (matches original async_main)"""
    print("\n--- Testing AsyncWebCrawler directly ---")

    from crawl4ai.async_configs import CrawlerRunConfig
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai import AsyncWebCrawler

    # Example: ignore all links, don't escape HTML, and wrap text at 80 characters
    md_generator = DefaultMarkdownGenerator(
        options={
            "ignore_links": True,
            "escape_html": False,
            "body_width": 80
        }
    )

    config = CrawlerRunConfig(markdown_generator=md_generator)

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url, config=config)
        if hasattr(result, 'success') and result.success:
            print("✅ Crawler successful!")
            print(f"Markdown length: {len(result.markdown)} characters")
            print("\nFirst 300 characters of markdown:")
            print("-" * 50)
            print(result.markdown[:300])
            print("-" * 50)
        else:
            print(f"❌ Crawler failed: {getattr(result, 'error_message', 'Unknown error')}")


async def sync_crawler_test_async(url):
    """Test wrapper function for crawl_website_sync, but called from async context"""
    print("\n--- Testing crawl_website_sync ---")

    # Get the current event loop
    loop = asyncio.get_event_loop()

    # Run the sync function in an executor to avoid blocking the event loop
    def run_sync_crawler():
        try:
            from web_crawler_lib import crawl_website

            # Instead of using crawl_website_sync which internally calls asyncio.run(),
            # we create a new loop and run the coroutine there
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)

            try:
                crawl_result = new_loop.run_until_complete(
                    crawl_website(
                        url=url,
                        excluded_tags=['form', 'header'],
                        verbose=True
                    )
                )
                return crawl_result
            finally:
                new_loop.close()

        except Exception as e:
            import traceback
            print(f"Error in sync crawler: {e}")
            print(traceback.format_exc())
            return WebCrawlerResult(success=False, error_message=str(e))

    # Run the sync function in an executor to avoid blocking
    crawl_result = await loop.run_in_executor(None, run_sync_crawler)

    if crawl_result.success:
        print("✅ Sync crawler successful!")
        print(f"Markdown length: {len(crawl_result.markdown)} characters")
        print("\nFirst 300 characters of markdown:")
        print("-" * 50)
        print(crawl_result.markdown[:10000])
        print("-" * 50)
    else:
        print(f"❌ Sync crawler failed: {crawl_result.error_message}")


async def crawl_website_direct_test(url):
    """Test our crawl_website function directly"""
    print("\n--- Testing crawl_website directly ---")

    from web_crawler_lib import crawl_website

    crawl_result = await crawl_website(
        url=url,
        excluded_tags=['form', 'header'],
        verbose=True
    )

    if crawl_result.success:
        print("✅ Direct async function successful!")
        print(f"Markdown length: {len(crawl_result.markdown)} characters")
        print("\nFirst 300 characters of markdown:")
        print("-" * 50)
        print(crawl_result.markdown[:300])
        print("-" * 50)
    else:
        print(f"❌ Direct async function failed: {crawl_result.error_message}")


async def run_all_tests(url):
    """Run all tests in sequence"""
    print(f"Testing URL: {url}")
    print(f"Python version: {sys.version}")

    # First test our sync wrapper (but called in async-compatible way)
    await sync_crawler_test_async(url)

    # Then test our async function directly
    await crawl_website_direct_test(url)

    # Finally test the original implementation
    await async_crawler_test(url)


def main():
    """Main function to test the crawler with different approaches"""
    import argparse

    parser = argparse.ArgumentParser(description='Web Crawler Debug Tool')
    parser.add_argument('--url', type=str,
                        default='https://matriq.co.za/job/boilermaker/',
                        help='URL to crawl')
    args = parser.parse_args()

    # Run all tests
    asyncio.run(run_all_tests(args.url))


if __name__ == "__main__":
    main()