#!/usr/bin/env python3
"""
Direct skills insertion test
"""
import sqlite3
from recruitment_db_lib import RecruitmentDatabase

# Create database instance
db = RecruitmentDatabase()

# URL ID to test (replace with an actual URL ID from your database)
url_id = 585  # Change to match one of your URL IDs

# Test skills from your logs
test_skills = [
    {"skill": "voice process", "experience": None},
    {"skill": "semi-voice process", "experience": None},
    {"skill": "customer service", "experience": None},
    {"skill": "operations", "experience": None}
]

print(f"Testing direct skill insertion for URL ID {url_id}")

# First check if skills already exist
try:
    query = "SELECT COUNT(*) FROM skills WHERE url_id = ?"
    with db._execute_query(query, (url_id,)) as cursor:
        count = cursor.fetchone()[0]
        print(f"URL ID {url_id} already has {count} skills")

        if count > 0:
            query = "SELECT skill, experience FROM skills WHERE url_id = ?"
            with db._execute_query(query, (url_id,)) as cursor:
                skills = cursor.fetchall()
                print("Existing skills:")
                for skill, exp in skills:
                    print(f"  {skill}: {exp}")
except Exception as e:
    print(f"Error checking existing skills: {e}")

# Direct insertion test
success_count = 0
for skill_data in test_skills:
    try:
        query = "INSERT OR IGNORE INTO skills (url_id, skill, experience) VALUES (?, ?, ?)"
        with db._execute_query(query, (url_id, skill_data["skill"], skill_data["experience"])) as cursor:
            if cursor.rowcount > 0:
                success_count += 1
                print(
                    f"Successfully inserted skill '{skill_data['skill']}' with experience '{skill_data['experience']}'")
            else:
                print(f"Failed to insert skill '{skill_data['skill']}' (possibly already exists)")
    except Exception as e:
        print(f"Error inserting skill '{skill_data['skill']}': {e}")

print(f"Successfully inserted {success_count} out of {len(test_skills)} skills")

# Verify skills were inserted
try:
    query = "SELECT skill, experience FROM skills WHERE url_id = ?"
    with db._execute_query(query, (url_id,)) as cursor:
        skills = cursor.fetchall()
        print(f"After insertion: Found {len(skills)} skills:")
        for skill, exp in skills:
            print(f"  {skill}: {exp}")
except Exception as e:
    print(f"Error verifying skills: {e}")