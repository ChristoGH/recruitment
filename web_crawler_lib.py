"""
Web Crawler Library

This module provides functions to crawl websites and extract their content
in a structured format.
"""

import asyncio
import sys
import traceback
from typing import Dict, Any, Optional, List, Union
from multiprocessing import Process
from queue import Queue, Empty

from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai import AsyncWebCrawler
from logging_config import setup_logging

# Create module-specific logger
logger = setup_logging("web_crawler_lib")


class WebCrawlerResult:
    """Class to hold the result of a web crawl operation."""

    def __init__(
            self,
            success: bool,
            markdown: str = "",
            text: str = "",
            media: Dict[str, List[Dict[str, str]]] = None,
            links: Dict[str, List[Dict[str, str]]] = None,
            error_message: str = "",
            url: str = ""
    ):
        self.success = success
        self.markdown = markdown
        self.text = text  # This may not be used in the actual library response
        self.media = media or {"images": []}
        self.links = links or {"internal": [], "external": []}
        self.error_message = error_message
        self.url = url


async def crawl_website(
        url: str,
        word_count_threshold: int = 10,
        excluded_tags: List[str] = None,
        exclude_external_links: bool = True,
        process_iframes: bool = True,
        remove_overlay_elements: bool = True,
        use_cache: bool = True,
        verbose: bool = False
) -> WebCrawlerResult:
    """
    Crawl a website and extract its content.

    Args:
        url: The URL to crawl
        word_count_threshold: Minimum number of words for a text block to be included
        excluded_tags: HTML tags to exclude from processing
        exclude_external_links: Whether to exclude external links
        process_iframes: Whether to process iframes
        remove_overlay_elements: Whether to remove overlay elements
        use_cache: Whether to use cached results if available
        verbose: Whether to output verbose logs

    Returns:
        WebCrawlerResult: An object containing the crawl results
    """
    if excluded_tags is None:
        excluded_tags = ['form', 'header']

    md_generator = DefaultMarkdownGenerator(
        options={
            "ignore_links": True,
            "escape_html": False,
            "body_width": 80
        }
    )

    browser_config = BrowserConfig(verbose=verbose)

    cache_mode = CacheMode.ENABLED if use_cache else CacheMode.DISABLED

    run_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        # Content filtering
        word_count_threshold=word_count_threshold,
        excluded_tags=excluded_tags,
        exclude_external_links=exclude_external_links,

        # Content processing
        process_iframes=process_iframes,
        remove_overlay_elements=remove_overlay_elements,

        # Cache control
        cache_mode=cache_mode
    )

    try:
        if verbose:
            logger.debug(f"Starting crawler for URL: {url}")

        async with AsyncWebCrawler(config=browser_config) as crawler:
            if verbose:
                logger.debug("AsyncWebCrawler context created")

            result = await crawler.arun(url=url, config=run_config)

            if verbose:
                logger.debug(f"Crawler run completed. Result type: {type(result)}")

            # Check if result has the expected attributes
            if not hasattr(result, 'success'):
                if verbose:
                    logger.debug(f"Result doesn't have 'success' attribute. Available attributes: {dir(result)}")

                # If it has markdown but no success flag, consider it successful
                if hasattr(result, 'markdown') and result.markdown:
                    if verbose:
                        logger.debug(f"Found markdown content of length: {len(result.markdown)}")
                    return WebCrawlerResult(
                        success=True,
                        markdown=result.markdown,
                        # Don't try to access text attribute as it might not exist
                        media=getattr(result, 'media', {"images": []}),
                        links=getattr(result, 'links', {"internal": [], "external": []}),
                        url=url
                    )
                else:
                    if verbose:
                        logger.debug("No markdown content found")
                    return WebCrawlerResult(
                        success=False,
                        error_message="No content found",
                        url=url
                    )

            # Handle successful case - note we're not accessing the text attribute
            if result.success:
                if verbose:
                    logger.debug(f"Crawler successful with markdown length: {len(result.markdown)}")
                return WebCrawlerResult(
                    success=True,
                    markdown=result.markdown,
                    # Use an empty text field to avoid AttributeError
                    text="",  # The crawler doesn't return a 'text' field
                    media=getattr(result, 'media', {"images": []}),
                    links=getattr(result, 'links', {"internal": [], "external": []}),
                    url=url
                )
            else:
                if verbose:
                    logger.warning(f"Crawler failed with error: {getattr(result, 'error_message', 'Unknown error')}")
                return WebCrawlerResult(
                    success=False,
                    error_message=getattr(result, 'error_message', 'Unknown error'),
                    url=url
                )
    except Exception as e:
        error_details = traceback.format_exc()
        if verbose:
            logger.error(f"Crawler exception: {str(e)}")
            logger.error(f"Traceback: {error_details}")

        return WebCrawlerResult(
            success=False,
            error_message=f"Crawl failed with exception: {str(e)}",
            url=url
        )


def _run_in_new_loop(coro, *args, **kwargs):
    """Run a coroutine in a new event loop in a new thread"""
    result = None
    exception = None

    def run_in_thread():
        nonlocal result, exception
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            # Don't set it as the current loop, just use it directly
            try:
                # Run the coroutine in the new loop
                result = loop.run_until_complete(coro(*args, **kwargs))
            finally:
                # Clean up the loop
                loop.close()
        except Exception as e:
            exception = e

    # Create and start the thread
    import threading
    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()

    if exception:
        raise exception
    return result


def crawl_website_sync(
        url: str,
        word_count_threshold: int = 10,
        excluded_tags: List[str] = None,
        exclude_external_links: bool = True,
        process_iframes: bool = True,
        remove_overlay_elements: bool = True,
        use_cache: bool = True,
        verbose: bool = False
) -> WebCrawlerResult:
    """
    Synchronous version of crawl_website.

    This function has the same parameters as crawl_website but can be
    called from synchronous code.
    """
    try:
        if verbose:
            logger.debug("Starting synchronous crawler...")

        # Check if we're in an event loop already
        try:
            loop = asyncio.get_event_loop()
            in_event_loop = loop.is_running()
        except RuntimeError:
            # No event loop in this thread
            in_event_loop = False

        if verbose:
            logger.debug(f"Current thread has running event loop: {in_event_loop}")

        if in_event_loop:
            # If we're in an event loop, run the crawler in a separate thread
            if verbose:
                logger.debug("Running in a new thread with a new event loop")

            # Create a thread to run the crawler
            import threading
            result = None
            exception = None

            def run_in_thread():
                nonlocal result, exception
                try:
                    # Create a new event loop for this thread
                    loop = asyncio.new_event_loop()
                    # Don't set it as the current loop, just use it directly
                    try:
                        # Run the coroutine in the new loop
                        result = loop.run_until_complete(
                            crawl_website(
                                url=url,
                                word_count_threshold=word_count_threshold,
                                excluded_tags=excluded_tags,
                                exclude_external_links=exclude_external_links,
                                process_iframes=process_iframes,
                                remove_overlay_elements=remove_overlay_elements,
                                use_cache=use_cache,
                                verbose=verbose
                            )
                        )
                    finally:
                        # Clean up the loop
                        loop.close()
                except Exception as e:
                    exception = e

            # Create and start the thread
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()

            if exception:
                raise exception
            return result
        else:
            # If we're not in an event loop, we can use asyncio.run
            if verbose:
                logger.debug("Using asyncio.run in the current thread")

            return asyncio.run(
                crawl_website(
                    url=url,
                    word_count_threshold=word_count_threshold,
                    excluded_tags=excluded_tags,
                    exclude_external_links=exclude_external_links,
                    process_iframes=process_iframes,
                    remove_overlay_elements=remove_overlay_elements,
                    use_cache=use_cache,
                    verbose=verbose
                )
            )
    except Exception as e:
        error_details = traceback.format_exc()
        if verbose:
            logger.error(f"Sync crawler exception: {str(e)}")
            logger.error(f"Traceback: {error_details}")

        return WebCrawlerResult(
            success=False,
            error_message=f"Synchronous crawl failed with exception: {str(e)}"
        )


def run_in_process(url: str, max_pages: int, max_depth: int, timeout: int, verbose: bool, queue: Queue) -> None:
    """Run the crawler in a separate process."""
    try:
        # Create a new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the crawler
        result = loop.run_until_complete(crawl_website(
            url=url,
            max_pages=max_pages,
            max_depth=max_depth,
            timeout=timeout,
            verbose=verbose
        ))
        
        # Put the result in the queue
        queue.put(result)
    except Exception as e:
        # Put the exception in the queue
        queue.put(e)
    finally:
        # Clean up
        try:
            loop.close()
        except:
            pass


def crawl_website_sync_v2(
    url: str,
    max_pages: int = 10,
    max_depth: int = 2,
    timeout: int = 30,
    verbose: bool = False
) -> WebCrawlerResult:
    """
    Synchronous version of crawl_website that runs in a separate process.
    This version is more reliable for handling event loop issues.
    """
    try:
        # Create a queue for the result
        result_queue = Queue()
        
        # Create and start the process
        process = Process(
            target=run_in_process,
            args=(url, max_pages, max_depth, timeout, verbose, result_queue)
        )
        process.start()
        
        # Wait for the result with timeout
        try:
            result = result_queue.get(timeout=timeout + 5)  # Add 5 seconds buffer
        except Empty:
            process.terminate()
            process.join()
            return WebCrawlerResult(
                success=False,
                url=url,
                error_message="Crawling timed out"
            )
        
        # Check if we got an exception
        if isinstance(result, Exception):
            if verbose:
                logger.error(f"Sync crawler exception (v2): {str(result)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
            return WebCrawlerResult(
                success=False,
                url=url,
                error_message=str(result)
            )
        
        return result
        
    except Exception as e:
        if verbose:
            logger.error(f"Sync crawler exception (v2): {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        return WebCrawlerResult(
            success=False,
            url=url,
            error_message=str(e)
        )