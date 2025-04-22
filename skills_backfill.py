#!/usr/bin/env python3
"""
Skills Debug and Backfill Script

This script helps diagnose skills insertion issues and backfill missing skills
for URLs that have already been processed.

Usage:
  python skills_debug.py [--backfill] [--url-id URL_ID]

Options:
  --backfill      Backfill missing skills for all recruitment URLs
  --url-id URL_ID Focus on a specific URL ID

  python skills_debug.py --url-id 580
"""

import argparse
import logging
import sys
from typing import Dict, List, Any, Optional
from recruitment_db import RecruitmentDatabase
from recruitment_models import transform_skills_response, SkillExperience

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('skills_debug.log')
    ]
)
logger = logging.getLogger("skills_debug")


def check_skills_table_structure():
    """Verify skills table structure and constraints"""
    db = RecruitmentDatabase()

    logger.info("Checking skills table structure...")

    # Check table info
    try:
        query = "PRAGMA table_info(skills)"
        with db._execute_query(query) as cursor:
            columns = cursor.fetchall()

        column_names = [col[1] for col in columns]
        logger.info(f"Skills table columns: {column_names}")

        # Check if necessary columns exist
        if 'skill' not in column_names:
            logger.error("Missing 'skill' column in skills table!")
        if 'experience' not in column_names:
            logger.error("Missing 'experience' column in skills table!")

        # Check table constraints
        query = "SELECT sql FROM sqlite_master WHERE type='table' AND name='skills'"
        with db._execute_query(query) as cursor:
            table_def = cursor.fetchone()
            if table_def:
                logger.info(f"Table definition: {table_def[0]}")

        # Count total records
        query = "SELECT COUNT(*) FROM skills"
        with db._execute_query(query) as cursor:
            count = cursor.fetchone()[0]
            logger.info(f"Total skills in database: {count}")

        # Sample some records
        query = "SELECT url_id, skill, experience FROM skills LIMIT 5"
        with db._execute_query(query) as cursor:
            samples = cursor.fetchall()
            if samples:
                logger.info("Sample skills records:")
                for url_id, skill, exp in samples:
                    logger.info(f"  URL ID {url_id}: {skill} - {exp}")
            else:
                logger.warning("No skills records found in database")

        return True
    except Exception as e:
        logger.error(f"Error checking skills table structure: {e}")
        return False


def find_urls_missing_skills():
    """Find URLs that are marked as recruitment but have no skills"""
    db = RecruitmentDatabase()

    try:
        # Find URLs with recruitment_flag=1 but no skills
        query = """
            SELECT u.id, u.url, u.domain_name FROM urls u
            LEFT JOIN (
                SELECT url_id, COUNT(*) as skill_count 
                FROM skills 
                GROUP BY url_id
            ) s ON u.id = s.url_id
            WHERE u.recruitment_flag = 1 
            AND (s.skill_count IS NULL OR s.skill_count = 0)
            ORDER BY u.id
        """

        with db._execute_query(query) as cursor:
            missing_skills_urls = cursor.fetchall()

        if missing_skills_urls:
            logger.info(f"Found {len(missing_skills_urls)} URLs missing skills:")
            for i, (url_id, url, domain) in enumerate(missing_skills_urls[:10]):  # Show first 10
                logger.info(f"  {i + 1}. URL ID {url_id}: {domain} - {url}")

            if len(missing_skills_urls) > 10:
                logger.info(f"  ...and {len(missing_skills_urls) - 10} more")
        else:
            logger.info("No URLs missing skills found")

        return missing_skills_urls
    except Exception as e:
        logger.error(f"Error finding URLs missing skills: {e}")
        return []


def test_skills_insertion(url_id):
    """Test inserting skills for a specific URL ID"""
    db = RecruitmentDatabase()
    logger.info(f"Testing skills insertion for URL ID {url_id}")

    # Define test skills based on the common format in your logs
    test_skills = [
        ("Test Skill 1", "1 year"),
        ("Test Skill 2", "3-5 years"),
        ("Test Skill 3", "not_listed"),
        ("Test Skill 4", None)
    ]

    # First check if this URL already has test skills
    try:
        query = "SELECT skill, experience FROM skills WHERE url_id = ? AND skill LIKE 'Test Skill%'"
        with db._execute_query(query, (url_id,)) as cursor:
            existing_test_skills = cursor.fetchall()

        if existing_test_skills:
            logger.info(f"Found {len(existing_test_skills)} existing test skills for URL ID {url_id}")
            for skill, exp in existing_test_skills:
                logger.info(f"  {skill}: {exp}")

            # Delete existing test skills
            logger.info(f"Deleting existing test skills for URL ID {url_id}")
            query = "DELETE FROM skills WHERE url_id = ? AND skill LIKE 'Test Skill%'"
            with db._execute_query(query, (url_id,)) as cursor:
                logger.info(f"Deleted {cursor.rowcount} test skills")
    except Exception as e:
        logger.error(f"Error checking/deleting existing test skills: {e}")

    # Insert skills directly with individual INSERT statements
    success_count = 0
    for skill, experience in test_skills:
        try:
            # Normalize "not_listed" to None
            if experience == "not_listed":
                experience = None

            query = "INSERT OR IGNORE INTO skills (url_id, skill, experience) VALUES (?, ?, ?)"
            with db._execute_query(query, (url_id, skill, experience)) as cursor:
                if cursor.rowcount > 0:
                    success_count += 1
                    logger.info(f"Successfully inserted skill '{skill}' with experience '{experience}'")
                else:
                    logger.warning(f"Skill '{skill}' not inserted (possibly already exists)")
        except Exception as e:
            logger.error(f"Error inserting skill '{skill}': {e}")

    logger.info(f"Inserted {success_count} out of {len(test_skills)} test skills")

    # Verify insertion
    try:
        query = "SELECT skill, experience FROM skills WHERE url_id = ? AND skill LIKE 'Test Skill%'"
        with db._execute_query(query, (url_id,)) as cursor:
            inserted_skills = cursor.fetchall()

        if inserted_skills:
            logger.info(f"Found {len(inserted_skills)} inserted test skills:")
            for skill, exp in inserted_skills:
                logger.info(f"  {skill}: {exp}")
        else:
            logger.error("No inserted test skills found - insertion failed!")
    except Exception as e:
        logger.error(f"Error verifying inserted skills: {e}")

    return success_count == len(test_skills)


def backfill_skills_from_logs(url_id=None):
    """Backfill skills for URLs from log data"""
    db = RecruitmentDatabase()

    # Get a mapping of URL IDs to tuples extracted from logs
    # You'll need to manually enter the skills data from your logs
    skills_from_logs = {
        # Example from your logs
        580: [
            ("Terminal Experience", "1 year"),
            ("Document control and management", "5 years"),
            ("Quality Assurance", None),  # Already converted from "not_listed"
            ("Instrumentation", "3 years"),
            ("Relevant experience", "2 years"),
            ("Information Systems, Computer Science or related field", None)
        ],
        # Add more URL ID/skills mappings as needed
    }

    if url_id:
        # Filter to just the requested URL ID
        if url_id in skills_from_logs:
            urls_to_process = {url_id: skills_from_logs[url_id]}
        else:
            logger.error(f"No skills data available for URL ID {url_id}")
            return False
    else:
        urls_to_process = skills_from_logs

    success_count = 0
    for url_id, skills in urls_to_process.items():
        logger.info(f"Backfilling {len(skills)} skills for URL ID {url_id}")

        # Check if this URL ID exists
        url_record = db.get_url_by_id(url_id)
        if not url_record:
            logger.error(f"URL ID {url_id} not found in database")
            continue

        # Check if URL already has skills
        query = "SELECT COUNT(*) FROM skills WHERE url_id = ?"
        with db._execute_query(query, (url_id,)) as cursor:
            count = cursor.fetchone()[0]

        if count > 0:
            logger.info(f"URL ID {url_id} already has {count} skills. Skipping.")
            continue

        # Insert skills one by one
        inserted_count = 0
        for skill, experience in skills:
            try:
                query = "INSERT OR IGNORE INTO skills (url_id, skill, experience) VALUES (?, ?, ?)"
                with db._execute_query(query, (url_id, skill, experience)) as cursor:
                    if cursor.rowcount > 0:
                        inserted_count += 1
                        logger.info(f"Inserted skill '{skill}' with experience '{experience}'")
                    else:
                        logger.warning(f"Skill '{skill}' not inserted (possibly duplicate)")
            except Exception as e:
                logger.error(f"Error inserting skill '{skill}': {e}")

        if inserted_count > 0:
            logger.info(f"Successfully backfilled {inserted_count} skills for URL ID {url_id}")
            success_count += 1

    return success_count > 0


def inspect_url_content_for_skills(url_id):
    """Manually inspect URL content to look for skills"""
    db = RecruitmentDatabase()

    try:
        url_record = db.get_url_by_id(url_id)
        if not url_record or not url_record.content:
            logger.error(f"No content found for URL ID {url_id}")
            return False

        # Print the first 1000 characters of content to help identify skills
        content = url_record.content[:1000]
        logger.info(f"URL ID {url_id} content sample (first 1000 chars):")
        logger.info(content)
        logger.info("...")

        # Look for skills-related keywords
        skills_keywords = ["skill", "skills", "experience", "years", "knowledge", "ability", "proficient",
                           "qualification"]

        logger.info(f"Searching for skills-related keywords:")
        for keyword in skills_keywords:
            if keyword.lower() in url_record.content.lower():
                # Print context around the keyword
                idx = url_record.content.lower().find(keyword.lower())
                start = max(0, idx - 50)
                end = min(len(url_record.content), idx + 50)
                context = url_record.content[start:end]
                logger.info(f"  Found '{keyword}': ...{context}...")

        return True
    except Exception as e:
        logger.error(f"Error inspecting URL content: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Debug and fix skills insertion issues")
    parser.add_argument("--backfill", action="store_true", help="Backfill missing skills")
    parser.add_argument("--url-id", type=int, help="Process specific URL ID")
    args = parser.parse_args()

    logger.info("Skills Debug and Backfill Script")

    # Check database structure
    if not check_skills_table_structure():
        logger.error("Skills table structure check failed. Exiting.")
        return

    # If URL ID is specified, focus on that URL
    if args.url_id:
        logger.info(f"Focusing on URL ID {args.url_id}")

        # Test skills insertion for this URL
        if test_skills_insertion(args.url_id):
            logger.info(f"Skills insertion test PASSED for URL ID {args.url_id}")
        else:
            logger.error(f"Skills insertion test FAILED for URL ID {args.url_id}")
            return

        # Inspect URL content
        logger.info(f"Inspecting content for URL ID {args.url_id}")
        inspect_url_content_for_skills(args.url_id)

        # Backfill if requested
        if args.backfill:
            logger.info(f"Attempting to backfill skills for URL ID {args.url_id}")
            backfill_skills_from_logs(args.url_id)
    else:
        # Find URLs missing skills
        missing_skills_urls = find_urls_missing_skills()

        if not missing_skills_urls:
            logger.info("No URLs missing skills found. Nothing to do.")
            return

        # Test skills insertion with the first URL
        if missing_skills_urls:
            first_url_id = missing_skills_urls[0][0]
            logger.info(f"Testing skills insertion with URL ID {first_url_id}")
            test_skills_insertion(first_url_id)

        # Backfill if requested
        if args.backfill:
            logger.info("Backfilling skills from logs")
            backfill_skills_from_logs()


if __name__ == "__main__":
    main()