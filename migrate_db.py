import sqlite3
import logging
from datetime import datetime
import os
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database paths
OLD_DB_PATH = "databases/recruitment.db"
NEW_DB_PATH = "databases/recruitment_new.db"

def connect_to_db(db_path):
    """Create a connection to the database."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database {db_path}: {e}")
        raise

def attach_new_db(conn, new_db_path):
    """Attach the new database."""
    try:
        cursor = conn.cursor()
        cursor.execute("ATTACH DATABASE ? AS new_db", (new_db_path,))
        conn.commit()
        logger.info("Attached new database")
    except sqlite3.Error as e:
        logger.error(f"Error attaching new database: {e}")
        raise

def create_agencies_table(new_conn):
    """Create the agencies table in the new database."""
    logger.info("Creating agencies table...")
    try:
        new_cursor = new_conn.cursor()
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS agencies (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        new_conn.commit()
        logger.info("Created agencies table")
    except sqlite3.Error as e:
        logger.error(f"Error creating agencies table: {e}")
        raise

def create_urls_table(new_conn):
    """Create the urls table in the new database."""
    logger.info("Creating urls table...")
    try:
        new_cursor = new_conn.cursor()
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS urls (
                id INTEGER PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                domain_name TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        new_conn.commit()
        logger.info("Created urls table")
    except sqlite3.Error as e:
        logger.error(f"Error creating urls table: {e}")
        raise

def create_companies_table(new_conn):
    """Create the companies table in the new database if it doesn't exist."""
    logger.info("Checking companies table...")
    try:
        new_cursor = new_conn.cursor()
        # Check if table exists
        new_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='companies'")
        if new_cursor.fetchone():
            logger.info("Companies table already exists, skipping creation")
            return
            
        # Create table only if it doesn't exist
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        new_conn.commit()
        logger.info("Created companies table")
    except sqlite3.Error as e:
        logger.error(f"Error with companies table: {e}")
        raise

def create_jobs_table(new_conn):
    """Create the jobs table in the new database."""
    logger.info("Creating jobs table...")
    try:
        new_cursor = new_conn.cursor()
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                salary_min DECIMAL,
                salary_max DECIMAL,
                salary_currency TEXT,
                status TEXT CHECK(status IN ('active', 'inactive', 'filled', 'expired', 'draft', 'published')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        new_conn.commit()
        logger.info("Created jobs table")
    except sqlite3.Error as e:
        logger.error(f"Error creating jobs table: {e}")
        raise

def create_adverts_table(new_conn):
    """Create the adverts table in the new database."""
    logger.info("Creating adverts table...")
    try:
        new_cursor = new_conn.cursor()
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS adverts (
                id INTEGER PRIMARY KEY,
                job_id INTEGER NOT NULL,
                posted_date DATE,
                application_deadline DATE,
                is_remote BOOLEAN DEFAULT FALSE,
                is_hybrid BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
            )
        """)
        new_conn.commit()
        logger.info("Created adverts table")
    except sqlite3.Error as e:
        logger.error(f"Error creating adverts table: {e}")
        raise

def create_skills_table(new_conn):
    """Create the skills table in the new database."""
    logger.info("Creating skills table...")
    try:
        new_cursor = new_conn.cursor()
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        new_conn.commit()
        logger.info("Created skills table")
    except sqlite3.Error as e:
        logger.error(f"Error creating skills table: {e}")
        raise

def create_job_skills_table(new_conn):
    """Create the job_skills table in the new database."""
    logger.info("Creating job_skills table...")
    try:
        new_cursor = new_conn.cursor()
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_skills (
                job_id INTEGER NOT NULL,
                skill_id INTEGER NOT NULL,
                experience TEXT CHECK(experience IN ('entry', 'intermediate', 'senior', 'expert')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (job_id, skill_id),
                FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
                FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE
            )
        """)
        new_conn.commit()
        logger.info("Created job_skills table")
    except sqlite3.Error as e:
        logger.error(f"Error creating job_skills table: {e}")
        raise

def create_qualifications_table(new_conn):
    """Create the qualifications table in the new database."""
    logger.info("Creating qualifications table...")
    try:
        new_cursor = new_conn.cursor()
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS qualifications (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        new_conn.commit()
        logger.info("Created qualifications table")
    except sqlite3.Error as e:
        logger.error(f"Error creating qualifications table: {e}")
        raise

def create_job_qualifications_table(new_conn):
    """Create the job_qualifications table in the new database."""
    logger.info("Creating job_qualifications table...")
    try:
        new_cursor = new_conn.cursor()
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_qualifications (
                job_id INTEGER NOT NULL,
                qualification_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (job_id, qualification_id),
                FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
                FOREIGN KEY (qualification_id) REFERENCES qualifications(id) ON DELETE CASCADE
            )
        """)
        new_conn.commit()
        logger.info("Created job_qualifications table")
    except sqlite3.Error as e:
        logger.error(f"Error creating job_qualifications table: {e}")
        raise

def create_attributes_table(new_conn):
    """Create the attributes table in the new database."""
    logger.info("Creating attributes table...")
    try:
        new_cursor = new_conn.cursor()
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS attributes (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        new_conn.commit()
        logger.info("Created attributes table")
    except sqlite3.Error as e:
        logger.error(f"Error creating attributes table: {e}")
        raise

def create_job_attributes_table(new_conn):
    """Create the job_attributes table in the new database."""
    logger.info("Creating job_attributes table...")
    try:
        new_cursor = new_conn.cursor()
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_attributes (
                job_id INTEGER NOT NULL,
                attribute_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (job_id, attribute_id),
                FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
                FOREIGN KEY (attribute_id) REFERENCES attributes(id) ON DELETE CASCADE
            )
        """)
        new_conn.commit()
        logger.info("Created job_attributes table")
    except sqlite3.Error as e:
        logger.error(f"Error creating job_attributes table: {e}")
        raise

def create_duties_table(new_conn):
    """Create the duties table in the new database."""
    logger.info("Creating duties table...")
    try:
        new_cursor = new_conn.cursor()
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS duties (
                id INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        new_conn.commit()
        logger.info("Created duties table")
    except sqlite3.Error as e:
        logger.error(f"Error creating duties table: {e}")
        raise

def create_job_duties_table(new_conn):
    """Create the job_duties table in the new database."""
    logger.info("Creating job_duties table...")
    try:
        new_cursor = new_conn.cursor()
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_duties (
                job_id INTEGER NOT NULL,
                duty_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (job_id, duty_id),
                FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
                FOREIGN KEY (duty_id) REFERENCES duties(id) ON DELETE CASCADE
            )
        """)
        new_conn.commit()
        logger.info("Created job_duties table")
    except sqlite3.Error as e:
        logger.error(f"Error creating job_duties table: {e}")
        raise

def create_locations_table(new_conn):
    """Create the locations table in the new database."""
    logger.info("Creating locations table...")
    try:
        new_cursor = new_conn.cursor()
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS locations (
                id INTEGER PRIMARY KEY,
                country TEXT NOT NULL,
                province TEXT,
                city TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        new_conn.commit()
        logger.info("Created locations table")
    except sqlite3.Error as e:
        logger.error(f"Error creating locations table: {e}")
        raise

def create_job_locations_table(new_conn):
    """Create the job_locations table in the new database."""
    logger.info("Creating job_locations table...")
    try:
        new_cursor = new_conn.cursor()
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_locations (
                job_id INTEGER NOT NULL,
                location_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (job_id, location_id),
                FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
                FOREIGN KEY (location_id) REFERENCES locations(id) ON DELETE CASCADE
            )
        """)
        new_conn.commit()
        logger.info("Created job_locations table")
    except sqlite3.Error as e:
        logger.error(f"Error creating job_locations table: {e}")
        raise

def create_phones_table(new_conn):
    """Create the phones table in the new database."""
    logger.info("Creating phones table...")
    try:
        new_cursor = new_conn.cursor()
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS phones (
                id INTEGER PRIMARY KEY,
                number TEXT NOT NULL,
                type TEXT CHECK(type IN ('mobile', 'landline', 'fax')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        new_conn.commit()
        logger.info("Created phones table")
    except sqlite3.Error as e:
        logger.error(f"Error creating phones table: {e}")
        raise

def create_phone_relationship_tables(new_conn):
    """Create the company_phones and agency_phones relationship tables."""
    logger.info("Creating phone relationship tables...")
    try:
        new_cursor = new_conn.cursor()
        
        # Create company_phones table
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS company_phones (
                company_id INTEGER NOT NULL,
                phone_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (company_id, phone_id),
                FOREIGN KEY (company_id) REFERENCES companies(id) ON DELETE CASCADE,
                FOREIGN KEY (phone_id) REFERENCES phones(id) ON DELETE CASCADE
            )
        """)
        
        # Create agency_phones table
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS agency_phones (
                agency_id INTEGER NOT NULL,
                phone_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (agency_id, phone_id),
                FOREIGN KEY (agency_id) REFERENCES agencies(id) ON DELETE CASCADE,
                FOREIGN KEY (phone_id) REFERENCES phones(id) ON DELETE CASCADE
            )
        """)
        
        new_conn.commit()
        logger.info("Created phone relationship tables")
    except sqlite3.Error as e:
        logger.error(f"Error creating phone relationship tables: {e}")
        raise

def create_emails_table(conn: sqlite3.Connection) -> None:
    """Create the emails table in the new database."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
        DROP TABLE IF EXISTS emails;
        """)
        cursor.execute("""
        CREATE TABLE emails (
            id INTEGER PRIMARY KEY,
            email TEXT NOT NULL,
            url_id INTEGER,
            type TEXT CHECK(type IN ('primary', 'secondary', 'work')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (url_id) REFERENCES urls(id) ON DELETE CASCADE
        );
        """)
        conn.commit()
        logging.info("Created emails table successfully")
    except sqlite3.Error as e:
        logging.error(f"Error creating emails table: {e}")
        raise

def create_benefits_table(new_conn):
    """Create the benefits table in the new database."""
    new_cursor = new_conn.cursor()
    new_cursor.execute("""
        CREATE TABLE IF NOT EXISTS benefits (
            id INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    new_conn.commit()
    logger.info("Created benefits table in new database")

def map_experience_level(experience):
    if not experience:
        return 'entry'  # Default to entry level if no experience specified
        
    experience = experience.lower()
    
    # Entry level
    if any(x in experience for x in ['entry level', '0-2', '0-6', '1 year', '1-2', '1+', '6 months']):
        return 'entry'
    
    # Intermediate level
    if any(x in experience for x in ['2 years', '2-3', '2+', '2-5', '3 years', '3-5', '3+']):
        return 'intermediate'
    
    # Senior level
    if any(x in experience for x in ['4-5', '4-6', '4+', '5 years', '5-7', '5-10', '5-plus', '6-8', '6 to 9']):
        return 'senior'
    
    # Expert level
    if any(x in experience for x in ['7 years', '8 years', '8 to 10', '10+', '12 years', '12-15', '14 years']):
        return 'expert'
    
    return 'entry'  # Default to entry level if no match found

def migrate_agencies(old_conn, new_conn):
    """Migrate agencies from old database to new database."""
    logger.info("Starting agencies migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get count of agencies in old database
        old_cursor.execute("SELECT COUNT(*) FROM agency")
        old_count = old_cursor.fetchone()[0]
        logger.info(f"Found {old_count} agencies in old database")
        
        # Get all unique agencies from old database
        old_cursor.execute("""
            SELECT DISTINCT TRIM(agency) as name
            FROM agency
            WHERE agency IS NOT NULL 
            AND TRIM(agency) != ''
        """)
        agencies = old_cursor.fetchall()
        
        # Insert agencies into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for agency in agencies:
            try:
                name = agency['name'].strip()
                if name:
                    new_cursor.execute("""
                        INSERT INTO agencies (name)
                        VALUES (?)
                    """, (name,))
                    successful += 1
                else:
                    skipped += 1
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                    logger.debug(f"Skipped duplicate agency: {name}")
                else:
                    errors += 1
                    logger.error(f"Error inserting agency '{name}': {e}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting agency '{name}': {e}")

        new_conn.commit()
        logger.info(f"Agencies migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates/empty): {skipped}")
        logger.info(f"- Errors: {errors}")
        
        # Verify the migration
        new_cursor.execute("SELECT COUNT(*) FROM agencies")
        final_count = new_cursor.fetchone()[0]
        logger.info(f"Total agencies in new database: {final_count}")
        
        return successful, skipped, errors
        
    except Exception as e:
        logger.error(f"Error during agencies migration: {e}")
        raise

def migrate_urls(old_conn, new_conn):
    """Migrate URLs from old database to new database."""
    logger.info("Starting URLs migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get all URLs from old database
        old_cursor.execute("""
            SELECT id, url, domain_name, source, extracted_date
            FROM urls 
            WHERE url IS NOT NULL 
            AND TRIM(url) != ''
        """)
        urls = old_cursor.fetchall()
        
        # Insert URLs into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for url_record in urls:
            try:
                url = url_record['url'].strip()
                if url:
                    created_at = url_record['extracted_date'] or datetime.now().isoformat()
                    
                    new_cursor.execute("""
                        INSERT INTO urls (id, url, domain_name, source, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        url_record['id'],
                        url,
                        url_record['domain_name'],
                        url_record['source'],
                        created_at,
                        created_at
                    ))
                    successful += 1
                else:
                    skipped += 1
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                    logger.debug(f"Skipped duplicate URL: {url}")
                else:
                    errors += 1
                    logger.error(f"Error inserting URL '{url}': {e}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting URL '{url}': {e}")

        new_conn.commit()
        logger.info(f"URLs migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates/empty): {skipped}")
        logger.info(f"- Errors: {errors}")
        
        # Verify the migration
        new_cursor.execute("SELECT COUNT(*) FROM urls")
        final_count = new_cursor.fetchone()[0]
        logger.info(f"Total URLs in new database: {final_count}")
        
        return successful, skipped, errors
        
    except Exception as e:
        logger.error(f"Error during URLs migration: {e}")
        raise

def migrate_companies(old_conn, new_conn):
    """Migrate companies from old database to new database.
    
    Companies are organizations hiring for positions, distinct from agencies
    which are recruitment firms that post jobs on behalf of companies.
    """
    logger.info("Starting companies migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get all unique companies from the company table
        old_cursor.execute("""
            SELECT DISTINCT TRIM(name) as name
            FROM company
            WHERE name IS NOT NULL 
            AND TRIM(name) != ''
        """)
        companies = old_cursor.fetchall()
        
        # Insert companies into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for company in companies:
            try:
                name = company['name'].strip()
                if name:
                    new_cursor.execute("""
                        INSERT INTO companies (name)
                        VALUES (?)
                    """, (name,))
                    successful += 1
                else:
                    skipped += 1
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                    logger.debug(f"Skipped duplicate company: {name}")
                else:
                    errors += 1
                    logger.error(f"Error inserting company '{name}': {e}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting company '{name}': {e}")

        new_conn.commit()
        logger.info(f"Companies migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates/empty): {skipped}")
        logger.info(f"- Errors: {errors}")
        
        # Verify the migration
        new_cursor.execute("SELECT COUNT(*) FROM companies")
        final_count = new_cursor.fetchone()[0]
        logger.info(f"Total companies in new database: {final_count}")
        
        return successful, skipped, errors
        
    except Exception as e:
        logger.error(f"Error during companies migration: {e}")
        raise

def migrate_jobs(old_conn, new_conn):
    """Migrate jobs from old database to new database."""
    logger.info("Starting jobs migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get count of jobs in old database
        old_cursor.execute("SELECT COUNT(*) FROM job_adverts")
        old_count = old_cursor.fetchone()[0]
        logger.info(f"Found {old_count} jobs in old database")
        
        # Get all jobs from old database
        old_cursor.execute("""
            SELECT ja.id, ja.title, u.content as description, ja.url_id, c.id as company_id, ja.recruiter_id
            FROM job_adverts ja
            LEFT JOIN urls u ON ja.url_id = u.id
            LEFT JOIN company c ON ja.url_id = c.url_id
            WHERE ja.title IS NOT NULL 
            AND TRIM(ja.title) != ''
        """)
        jobs = old_cursor.fetchall()
        
        # Log sample jobs for debugging
        for i in range(min(5, len(jobs))):
            job = jobs[i]
            logger.debug(f"Sample job {i+1}:")
            logger.debug(f"- ID: {job['id']}")
            logger.debug(f"- Title: {job['title']}")
            logger.debug(f"- Description: {job['description'][:100]}...")
            logger.debug(f"- URL ID: {job['url_id']}")
            logger.debug(f"- Company ID: {job['company_id']}")
            logger.debug(f"- Recruiter ID: {job['recruiter_id']}")
        
        # Insert jobs into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for job in jobs:
            try:
                title = job['title'].strip() if job['title'] else None
                if not title:
                    logger.warning(f"Skipping job with empty title (ID: {job['id']})")
                    skipped += 1
                    continue
                
                # Insert job
                new_cursor.execute("""
                    INSERT INTO jobs (id, title, description, status)
                    VALUES (?, ?, ?, 'active')
                """, (job['id'], title, job['description']))
                
                # Insert job-URL relationship if URL exists
                if job['url_id']:
                    try:
                        new_cursor.execute("""
                            INSERT INTO job_urls (job_id, url_id)
                            VALUES (?, ?)
                        """, (job['id'], job['url_id']))
                    except sqlite3.Error as e:
                        logger.error(f"Error inserting job-URL relationship for job {job['id']}: {e}")
                
                # Insert job-company relationship if company exists
                if job['company_id']:
                    try:
                        new_cursor.execute("""
                            INSERT INTO job_companies (job_id, company_id)
                            VALUES (?, ?)
                        """, (job['id'], job['company_id']))
                    except sqlite3.Error as e:
                        logger.error(f"Error inserting job-company relationship for job {job['id']}: {e}")
                
                successful += 1
                
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                    logger.debug(f"Skipped duplicate job: {title}")
                else:
                    errors += 1
                    logger.error(f"Error inserting job (ID: {job['id']}, Title: {title}): {e}")
            except Exception as e:
                errors += 1
                logger.error(f"Error inserting job (ID: {job['id']}, Title: {title}): {e}")

        new_conn.commit()
        logger.info(f"Jobs migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates/empty): {skipped}")
        logger.info(f"- Errors: {errors}")
        
        # Verify the migration
        new_cursor.execute("SELECT COUNT(*) FROM jobs")
        final_count = new_cursor.fetchone()[0]
        logger.info(f"Total jobs in new database: {final_count}")
        
        return successful, skipped, errors
        
    except Exception as e:
        logger.error(f"Error during jobs migration: {e}")
        raise

def migrate_adverts(old_conn, new_conn):
    """Migrate job advertisements from old database to new database."""
    logger.info("Starting adverts migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get count of adverts in old database
        old_cursor.execute("SELECT COUNT(*) FROM job_advert_forms")
        old_count = old_cursor.fetchone()[0]
        logger.info(f"Found {old_count} adverts in old database")
        
        # Get all adverts from old database with correct join to new jobs
        old_cursor.execute("""
            SELECT 
                jaf.id,
                jaf.url_id,
                jaf.posted_date,
                jaf.application_deadline,
                ja.id as old_job_id
            FROM job_advert_forms jaf
            JOIN job_adverts ja ON jaf.url_id = ja.url_id
        """)
        adverts = old_cursor.fetchall()
        logger.info(f"Query returned {len(adverts)} adverts to migrate")
        
        # Insert adverts into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for advert in adverts:
            try:
                # The job_id in the new database is the same as the old one since we preserved IDs
                new_job_id = advert['old_job_id']
                
                # Insert into new adverts table
                new_cursor.execute("""
                    INSERT INTO adverts (
                        job_id,
                        posted_date,
                        application_deadline,
                        is_remote,
                        is_hybrid,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (
                    new_job_id,
                    advert['posted_date'],
                    advert['application_deadline'],
                    False,  # Default is_remote to False
                    False,  # Default is_hybrid to False
                ))
                
                successful += 1
                if successful % 100 == 0:
                    logger.info(f"Migrated {successful} adverts...")
                
            except sqlite3.Error as e:
                logger.error(f"Error migrating advert {advert['id']}: {e}")
                errors += 1
                continue
        
        new_conn.commit()
        logger.info(f"Adverts migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped: {skipped}")
        logger.info(f"- Errors: {errors}")
        
        # Verify the migration
        new_cursor.execute("SELECT COUNT(*) FROM adverts")
        final_count = new_cursor.fetchone()[0]
        logger.info(f"Total adverts in new database: {final_count}")
        
        return successful, skipped, errors
        
    except Exception as e:
        logger.error(f"Error during adverts migration: {e}")
        raise

def migrate_skills(old_conn, new_conn):
    """Migrate skills from old database to new database."""
    logger.info("Starting skills migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get count of skills in old database
        old_cursor.execute("SELECT COUNT(*) FROM skills")
        old_count = old_cursor.fetchone()[0]
        logger.info(f"Found {old_count} skills in old database")
        
        # Get all unique skills from old database
        old_cursor.execute("""
            SELECT DISTINCT TRIM(skill) as name
            FROM skills
            WHERE skill IS NOT NULL 
            AND TRIM(skill) != ''
        """)
        skills = old_cursor.fetchall()
        
        # Insert skills into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for skill in skills:
            try:
                name = skill['name'].strip()
                if name:
                    new_cursor.execute("""
                        INSERT INTO skills (name)
                        VALUES (?)
                    """, (name,))
                    successful += 1
                else:
                    skipped += 1
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                    logger.debug(f"Skipped duplicate skill: {name}")
                else:
                    errors += 1
                    logger.error(f"Error inserting skill '{name}': {e}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting skill '{name}': {e}")

        new_conn.commit()
        logger.info(f"Skills migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates/empty): {skipped}")
        logger.info(f"- Errors: {errors}")
        
        # Verify the migration
        new_cursor.execute("SELECT COUNT(*) FROM skills")
        final_count = new_cursor.fetchone()[0]
        logger.info(f"Total skills in new database: {final_count}")
        
        return successful, skipped, errors
        
    except Exception as e:
        logger.error(f"Error during skills migration: {e}")
        raise

def migrate_job_skills(old_conn, new_conn):
    """Migrate job skills from old database to new database."""
    logger.info("Starting job skills migration...")
    
    # Get all skills from old database
    old_cursor = old_conn.cursor()
    old_cursor.execute("""
        SELECT DISTINCT ja.id as job_id, s.skill, s.experience
        FROM skills s
        JOIN job_adverts ja ON s.url_id = ja.url_id
    """)
    skills = old_cursor.fetchall()
    
    logger.info(f"Found {len(skills)} job-skill relationships to migrate")
    
    # Insert skills into new database
    new_cursor = new_conn.cursor()
    migrated = 0
    skipped = 0
    errors = 0
    
    for skill in skills:
        job_id, skill_name, experience = skill
        
        try:
            # Get or create skill
            new_cursor.execute("SELECT id FROM skills WHERE name = ?", (skill_name,))
            skill_result = new_cursor.fetchone()
            if not skill_result:
                new_cursor.execute("INSERT INTO skills (name) VALUES (?)", (skill_name,))
                new_skill_id = new_cursor.lastrowid
            else:
                new_skill_id = skill_result[0]
            
            # Map experience to new categories
            mapped_experience = map_experience_level(experience)
            
            # Insert job-skill relationship
            new_cursor.execute("""
                INSERT INTO job_skills (job_id, skill_id, experience)
                VALUES (?, ?, ?)
            """, (job_id, new_skill_id, mapped_experience))
            
            migrated += 1
            if migrated % 100 == 0:
                logger.info(f"Migrated {migrated} job-skill relationships...")
                
        except Exception as e:
            logger.error(f"Error inserting job-skill relationship (Job {job_id}, Skill {skill_name}): {str(e)}")
            errors += 1
            continue
    
    new_conn.commit()
    
    # Get final count
    new_cursor.execute("SELECT COUNT(*) FROM job_skills")
    final_count = new_cursor.fetchone()[0]
    
    logger.info("Job skills migration completed:")
    logger.info(f"- Successfully migrated: {migrated}")
    logger.info(f"- Skipped (duplicates/not found): {skipped}")
    logger.info(f"- Errors: {errors}")
    logger.info(f"Total job-skill relationships in new database: {final_count}")

def migrate_qualifications(old_conn, new_conn):
    """Migrate qualifications from old database to new database."""
    logger.info("Starting qualifications migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get count of qualifications in old database
        old_cursor.execute("SELECT COUNT(*) FROM qualifications")
        old_count = old_cursor.fetchone()[0]
        logger.info(f"Found {old_count} qualifications in old database")
        
        # Get all unique qualifications from old database
        old_cursor.execute("""
            SELECT DISTINCT TRIM(qualification) as name
            FROM qualifications
            WHERE qualification IS NOT NULL 
            AND TRIM(qualification) != ''
        """)
        qualifications = old_cursor.fetchall()
        
        # Insert qualifications into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for qualification in qualifications:
            try:
                name = qualification['name'].strip()
                if name:
                    new_cursor.execute("""
                        INSERT INTO qualifications (name)
                        VALUES (?)
                    """, (name,))
                    successful += 1
                else:
                    skipped += 1
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                    logger.debug(f"Skipped duplicate qualification: {name}")
                else:
                    errors += 1
                    logger.error(f"Error inserting qualification '{name}': {e}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting qualification '{name}': {e}")

        new_conn.commit()
        logger.info(f"Qualifications migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates/empty): {skipped}")
        logger.info(f"- Errors: {errors}")
        
        # Verify the migration
        new_cursor.execute("SELECT COUNT(*) FROM qualifications")
        final_count = new_cursor.fetchone()[0]
        logger.info(f"Total qualifications in new database: {final_count}")
        
        return successful, skipped, errors
        
    except Exception as e:
        logger.error(f"Error during qualifications migration: {e}")
        raise

def migrate_job_qualifications(old_conn, new_conn):
    """Migrate job qualifications from old database to new database."""
    logger.info("Starting job qualifications migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get all job-qualification relationships from old database
        old_cursor.execute("""
            SELECT DISTINCT ja.id as job_id, q.qualification
            FROM qualifications q
            JOIN job_adverts ja ON q.url_id = ja.url_id
            WHERE q.qualification IS NOT NULL 
            AND TRIM(q.qualification) != ''
        """)
        job_qualifications = old_cursor.fetchall()
        
        logger.info(f"Found {len(job_qualifications)} job-qualification relationships to migrate")
        
        # Insert job-qualification relationships into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for job_qual in job_qualifications:
            try:
                job_id = job_qual['job_id']
                qualification_name = job_qual['qualification'].strip()
                
                # Get qualification ID from new database
                new_cursor.execute("""
                    SELECT id FROM qualifications 
                    WHERE name = ?
                """, (qualification_name,))
                qualification_result = new_cursor.fetchone()
                
                if qualification_result:
                    qualification_id = qualification_result[0]
                    
                    # Insert job-qualification relationship
                    new_cursor.execute("""
                        INSERT INTO job_qualifications (job_id, qualification_id)
                        VALUES (?, ?)
                    """, (job_id, qualification_id))
                    
                    successful += 1
                    if successful % 100 == 0:
                        logger.info(f"Migrated {successful} job-qualification relationships...")
                else:
                    skipped += 1
                    logger.debug(f"Qualification not found in new database: {qualification_name}")
                    
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                    logger.debug(f"Skipped duplicate job-qualification relationship")
                else:
                    errors += 1
                    logger.error(f"Error inserting job-qualification relationship: {e}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting job-qualification relationship: {e}")

        new_conn.commit()
        logger.info(f"Job qualifications migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates/not found): {skipped}")
        logger.info(f"- Errors: {errors}")
        
        # Verify the migration
        new_cursor.execute("SELECT COUNT(*) FROM job_qualifications")
        final_count = new_cursor.fetchone()[0]
        logger.info(f"Total job-qualification relationships in new database: {final_count}")
        
        return successful, skipped, errors
        
    except Exception as e:
        logger.error(f"Error during job qualifications migration: {e}")
        raise

def migrate_attributes(old_conn, new_conn):
    """Migrate attributes from old database to new database."""
    logger.info("Starting attributes migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get count of attributes in old database
        old_cursor.execute("SELECT COUNT(*) FROM attributes")
        old_count = old_cursor.fetchone()[0]
        logger.info(f"Found {old_count} attributes in old database")
        
        # Get all unique attributes from old database
        old_cursor.execute("""
            SELECT DISTINCT TRIM(attribute) as name
            FROM attributes
            WHERE attribute IS NOT NULL 
            AND TRIM(attribute) != ''
        """)
        attributes = old_cursor.fetchall()
        
        # Insert attributes into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for attribute in attributes:
            try:
                name = attribute['name'].strip()
                if name:
                    new_cursor.execute("""
                        INSERT INTO attributes (name)
                        VALUES (?)
                    """, (name,))
                    successful += 1
                else:
                    skipped += 1
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                    logger.debug(f"Skipped duplicate attribute: {name}")
                else:
                    errors += 1
                    logger.error(f"Error inserting attribute '{name}': {e}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting attribute '{name}': {e}")

        new_conn.commit()
        logger.info(f"Attributes migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates/empty): {skipped}")
        logger.info(f"- Errors: {errors}")
        
        # Verify the migration
        new_cursor.execute("SELECT COUNT(*) FROM attributes")
        final_count = new_cursor.fetchone()[0]
        logger.info(f"Total attributes in new database: {final_count}")
        
        return successful, skipped, errors
        
    except Exception as e:
        logger.error(f"Error during attributes migration: {e}")
        raise

def migrate_job_attributes(old_conn, new_conn):
    """Migrate job attributes from old database to new database."""
    logger.info("Starting job attributes migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get all job-attribute relationships from old database
        old_cursor.execute("""
            SELECT DISTINCT ja.id as job_id, a.attribute
            FROM attributes a
            JOIN job_adverts ja ON a.url_id = ja.url_id
            WHERE a.attribute IS NOT NULL 
            AND TRIM(a.attribute) != ''
        """)
        job_attributes = old_cursor.fetchall()
        
        logger.info(f"Found {len(job_attributes)} job-attribute relationships to migrate")
        
        # Insert job-attribute relationships into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for job_attr in job_attributes:
            try:
                job_id = job_attr['job_id']
                attribute_name = job_attr['attribute'].strip()
                
                # Get attribute ID from new database
                new_cursor.execute("""
                    SELECT id FROM attributes 
                    WHERE name = ?
                """, (attribute_name,))
                attribute_result = new_cursor.fetchone()
                
                if attribute_result:
                    attribute_id = attribute_result[0]
                    
                    # Insert job-attribute relationship
                    new_cursor.execute("""
                        INSERT INTO job_attributes (job_id, attribute_id)
                        VALUES (?, ?)
                    """, (job_id, attribute_id))
                    
                    successful += 1
                    if successful % 100 == 0:
                        logger.info(f"Migrated {successful} job-attribute relationships...")
                else:
                    skipped += 1
                    logger.debug(f"Attribute not found in new database: {attribute_name}")
                    
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                    logger.debug(f"Skipped duplicate job-attribute relationship")
                else:
                    errors += 1
                    logger.error(f"Error inserting job-attribute relationship: {e}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting job-attribute relationship: {e}")

        new_conn.commit()
        logger.info(f"Job attributes migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates/not found): {skipped}")
        logger.info(f"- Errors: {errors}")
        
        # Verify the migration
        new_cursor.execute("SELECT COUNT(*) FROM job_attributes")
        final_count = new_cursor.fetchone()[0]
        logger.info(f"Total job-attribute relationships in new database: {final_count}")
        
        return successful, skipped, errors
        
    except Exception as e:
        logger.error(f"Error during job attributes migration: {e}")
        raise

def migrate_duties(old_conn, new_conn):
    """Migrate duties from old database to new database."""
    logger.info("Starting duties migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get count of duties in old database
        old_cursor.execute("SELECT COUNT(*) FROM duties")
        old_count = old_cursor.fetchone()[0]
        logger.info(f"Found {old_count} duties in old database")
        
        # Get all unique duties from old database
        old_cursor.execute("""
            SELECT DISTINCT TRIM(duty) as description
            FROM duties
            WHERE duty IS NOT NULL 
            AND TRIM(duty) != ''
        """)
        duties = old_cursor.fetchall()
        
        # Insert duties into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for duty in duties:
            try:
                description = duty['description'].strip()
                if description:
                    new_cursor.execute("""
                        INSERT INTO duties (description)
                        VALUES (?)
                    """, (description,))
                    successful += 1
                else:
                    skipped += 1
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                    logger.debug(f"Skipped duplicate duty: {description}")
                else:
                    errors += 1
                    logger.error(f"Error inserting duty '{description}': {e}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting duty '{description}': {e}")

        new_conn.commit()
        logger.info(f"Duties migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates/empty): {skipped}")
        logger.info(f"- Errors: {errors}")
        
    except Exception as e:
        logger.error(f"Error during duties migration: {e}")
        raise

def migrate_job_duties(old_conn, new_conn):
    """Migrate job_duties relationships from old database to new database."""
    logger.info("Starting job_duties migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get all job_duties relationships from old database
        old_cursor.execute("""
            SELECT DISTINCT j.id as job_id, d.id as duty_id
            FROM job_adverts j
            JOIN duties d ON j.url_id = d.url_id
            WHERE j.id IS NOT NULL AND d.id IS NOT NULL
        """)
        relationships = old_cursor.fetchall()
        
        # Insert relationships into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for rel in relationships:
            try:
                new_cursor.execute("""
                    INSERT INTO job_duties (job_id, duty_id)
                    VALUES (?, ?)
                """, (rel['job_id'], rel['duty_id']))
                successful += 1
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                    logger.debug(f"Skipped duplicate job_duty relationship: {rel}")
                else:
                    errors += 1
                    logger.error(f"Error inserting job_duty relationship: {rel}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting job_duty relationship: {rel}")

        new_conn.commit()
        logger.info(f"Job duties migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates): {skipped}")
        logger.info(f"- Errors: {errors}")
        
    except Exception as e:
        logger.error(f"Error during job_duties migration: {e}")
        raise

def migrate_locations(old_conn, new_conn):
    """Migrate locations from old database to new database."""
    logger.info("Starting locations migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get count of locations in old database
        old_cursor.execute("SELECT COUNT(*) FROM location")
        old_count = old_cursor.fetchone()[0]
        logger.info(f"Found {old_count} locations in old database")
        
        # Get all unique locations from old database
        old_cursor.execute("""
            SELECT DISTINCT 
                TRIM(country) as country,
                TRIM(province) as province,
                TRIM(city) as city
            FROM location
            WHERE country IS NOT NULL 
            AND TRIM(country) != ''
        """)
        locations = old_cursor.fetchall()
        
        # Insert locations into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for location in locations:
            try:
                country = location['country'].strip()
                province = location['province'].strip() if location['province'] else None
                city = location['city'].strip() if location['city'] else None
                
                if country:
                    new_cursor.execute("""
                        INSERT INTO locations (country, province, city)
                        VALUES (?, ?, ?)
                    """, (country, province, city))
                    successful += 1
                else:
                    skipped += 1
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                    logger.debug(f"Skipped duplicate location: {country}, {province}, {city}")
                else:
                    errors += 1
                    logger.error(f"Error inserting location '{country}, {province}, {city}': {e}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting location '{country}, {province}, {city}': {e}")

        new_conn.commit()
        logger.info(f"Locations migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates/empty): {skipped}")
        logger.info(f"- Errors: {errors}")
        
        # Verify the migration
        new_cursor.execute("SELECT COUNT(*) FROM locations")
        final_count = new_cursor.fetchone()[0]
        logger.info(f"Total locations in new database: {final_count}")
        
        return successful, skipped, errors
        
    except Exception as e:
        logger.error(f"Error during locations migration: {e}")
        raise

def migrate_job_locations(old_conn, new_conn):
    """Migrate job locations from old database to new database."""
    logger.info("Starting job locations migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get all job-location relationships from old database
        old_cursor.execute("""
            SELECT DISTINCT ja.id as job_id, l.id as location_id
            FROM job_adverts ja
            JOIN location l ON ja.url_id = l.url_id
            WHERE ja.id IS NOT NULL AND l.id IS NOT NULL
        """)
        relationships = old_cursor.fetchall()
        
        logger.info(f"Found {len(relationships)} job-location relationships to migrate")
        
        # Insert relationships into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for rel in relationships:
            try:
                new_cursor.execute("""
                    INSERT INTO job_locations (job_id, location_id)
                    VALUES (?, ?)
                """, (rel['job_id'], rel['location_id']))
                successful += 1
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                    logger.debug(f"Skipped duplicate job-location relationship: {rel}")
                else:
                    errors += 1
                    logger.error(f"Error inserting job-location relationship: {rel}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting job-location relationship: {rel}")

        new_conn.commit()
        logger.info(f"Job locations migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates): {skipped}")
        logger.info(f"- Errors: {errors}")
        
        # Verify the migration
        new_cursor.execute("SELECT COUNT(*) FROM job_locations")
        final_count = new_cursor.fetchone()[0]
        logger.info(f"Total job-location relationships in new database: {final_count}")
        
        return successful, skipped, errors
        
    except Exception as e:
        logger.error(f"Error during job locations migration: {e}")
        raise

def verify_agencies_migration(old_conn, new_conn):
    """Verify the agencies migration."""
    logger.info("Verifying agencies migration...")
    
    old_cursor = old_conn.cursor()
    new_cursor = new_conn.cursor()
    
    # Get unique agency names from old database
    old_cursor.execute("""
        SELECT DISTINCT TRIM(name) as name
        FROM (
            SELECT agency as name FROM agency WHERE agency IS NOT NULL
            UNION
            SELECT name FROM recruiter WHERE name IS NOT NULL
        )
        WHERE TRIM(name) != ''
    """)
    old_agencies = set(row['name'].strip() for row in old_cursor.fetchall())
    
    # Get agency names from new database
    new_cursor.execute("SELECT name FROM agencies")
    new_agencies = set(row['name'] for row in new_cursor.fetchall())
    
    # Compare
    missing_agencies = old_agencies - new_agencies
    extra_agencies = new_agencies - old_agencies
    
    logger.info(f"Verification results:")
    logger.info(f"- Agencies in old database: {len(old_agencies)}")
    logger.info(f"- Agencies in new database: {len(new_agencies)}")
    
    if missing_agencies:
        logger.warning(f"Missing agencies: {missing_agencies}")
    if extra_agencies:
        logger.warning(f"Extra agencies in new database: {extra_agencies}")
    
    return len(missing_agencies) == 0 and len(extra_agencies) == 0

def migrate_phones(old_conn, new_conn):
    """Migrate phones from old database to new database."""
    logger.info("Starting phones migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get count of phones in old database
        old_cursor.execute("SELECT COUNT(*) FROM company_phone")
        old_count = old_cursor.fetchone()[0]
        logger.info(f"Found {old_count} phones in old database")
        
        # Get all unique phones from old database
        old_cursor.execute("""
            SELECT DISTINCT TRIM(phone) as number
            FROM company_phone
            WHERE phone IS NOT NULL 
            AND TRIM(phone) != ''
        """)
        phones = old_cursor.fetchall()
        
        # Insert phones into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for phone in phones:
            try:
                number = phone['number'].strip()
                if number:
                    # Try to determine phone type based on number format
                    phone_type = 'landline'  # default type
                    if number.startswith('+') or number.startswith('0'):
                        phone_type = 'mobile'
                    
                    new_cursor.execute("""
                        INSERT INTO phones (number, type)
                        VALUES (?, ?)
                    """, (number, phone_type))
                    successful += 1
                else:
                    skipped += 1
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                    logger.debug(f"Skipped duplicate phone: {number}")
                else:
                    errors += 1
                    logger.error(f"Error inserting phone '{number}': {e}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting phone '{number}': {e}")

        new_conn.commit()
        logger.info(f"Phones migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates/empty): {skipped}")
        logger.info(f"- Errors: {errors}")
        
        # Verify the migration
        new_cursor.execute("SELECT COUNT(*) FROM phones")
        final_count = new_cursor.fetchone()[0]
        logger.info(f"Total phones in new database: {final_count}")
        
        return successful, skipped, errors
        
    except Exception as e:
        logger.error(f"Error during phones migration: {e}")
        raise

def migrate_phone_relationships(old_conn, new_conn):
    """Migrate phone relationships from old database to new database."""
    logger.info("Starting phone relationships migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get company-phone relationships
        old_cursor.execute("""
            SELECT DISTINCT c.id as company_id, cp.phone
            FROM company c
            JOIN company_phone cp ON c.url_id = cp.url_id
            WHERE c.id IS NOT NULL AND cp.phone IS NOT NULL
        """)
        company_phones = old_cursor.fetchall()
        
        # Get agency-phone relationships
        old_cursor.execute("""
            SELECT DISTINCT a.id as agency_id, cp.phone
            FROM agency a
            JOIN company_phone cp ON a.url_id = cp.url_id
            WHERE a.id IS NOT NULL AND cp.phone IS NOT NULL
        """)
        agency_phones = old_cursor.fetchall()
        
        logger.info(f"Found {len(company_phones)} company-phone relationships to migrate")
        logger.info(f"Found {len(agency_phones)} agency-phone relationships to migrate")
        
        # Insert relationships into new database
        successful = 0
        skipped = 0
        errors = 0
        
        # Migrate company-phone relationships
        for rel in company_phones:
            try:
                company_id = rel['company_id']
                phone_number = rel['phone'].strip()
                
                # Get phone_id from new database
                new_cursor.execute("""
                    SELECT id FROM phones 
                    WHERE number = ?
                """, (phone_number,))
                phone_result = new_cursor.fetchone()
                
                if phone_result:
                    phone_id = phone_result[0]
                    new_cursor.execute("""
                        INSERT INTO company_phones (company_id, phone_id)
                        VALUES (?, ?)
                    """, (company_id, phone_id))
                    successful += 1
                else:
                    skipped += 1
                    logger.debug(f"Skipped company-phone relationship: company {company_id}, phone {phone_number}")
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                else:
                    errors += 1
                    logger.error(f"Error inserting company-phone relationship: {e}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting company-phone relationship: {e}")
        
        # Migrate agency-phone relationships
        for rel in agency_phones:
            try:
                agency_id = rel['agency_id']
                phone_number = rel['phone'].strip()
                
                # Get phone_id from new database
                new_cursor.execute("""
                    SELECT id FROM phones 
                    WHERE number = ?
                """, (phone_number,))
                phone_result = new_cursor.fetchone()
                
                if phone_result:
                    phone_id = phone_result[0]
                    new_cursor.execute("""
                        INSERT INTO agency_phones (agency_id, phone_id)
                        VALUES (?, ?)
                    """, (agency_id, phone_id))
                    successful += 1
                else:
                    skipped += 1
                    logger.debug(f"Skipped agency-phone relationship: agency {agency_id}, phone {phone_number}")
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                else:
                    errors += 1
                    logger.error(f"Error inserting agency-phone relationship: {e}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting agency-phone relationship: {e}")

        new_conn.commit()
        logger.info(f"Phone relationships migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates/not found): {skipped}")
        logger.info(f"- Errors: {errors}")
        
        # Verify the migration
        new_cursor.execute("SELECT COUNT(*) FROM company_phones")
        company_phones_count = new_cursor.fetchone()[0]
        new_cursor.execute("SELECT COUNT(*) FROM agency_phones")
        agency_phones_count = new_cursor.fetchone()[0]
        logger.info(f"Total company-phone relationships in new database: {company_phones_count}")
        logger.info(f"Total agency-phone relationships in new database: {agency_phones_count}")
        
        return successful, skipped, errors
        
    except Exception as e:
        logger.error(f"Error during phone relationships migration: {e}")
        raise

def migrate_emails(old_conn: sqlite3.Connection, new_conn: sqlite3.Connection) -> None:
    """Migrate emails from the old database to the new one."""
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()

        # Get all emails from all three tables in old database
        old_cursor.execute("""
            SELECT id, url_id, email, 'primary' as type FROM email
            UNION ALL
            SELECT id, url_id, email, 'work' as type FROM company_email
            UNION ALL
            SELECT id, url_id, email, 'secondary' as type FROM emails
            WHERE email IS NOT NULL AND TRIM(email) != ''
        """)
        emails = old_cursor.fetchall()
        logging.info(f"Found {len(emails)} emails to migrate")

        success_count = 0
        skip_count = 0
        error_count = 0

        for email_id, url_id, email, email_type in emails:
            try:
                # Clean the email
                email = email.strip()
                if not email:
                    skip_count += 1
                    continue

                # Check if email already exists
                new_cursor.execute("SELECT id FROM emails WHERE email = ? AND url_id = ?", (email, url_id))
                if new_cursor.fetchone() is not None:
                    skip_count += 1
                    continue

                # Insert email into new database
                new_cursor.execute("""
                    INSERT INTO emails (email, url_id, type)
                    VALUES (?, ?, ?)
                """, (email, url_id, email_type))
                success_count += 1

            except sqlite3.Error as e:
                logging.error(f"Error migrating email {email_id}: {e}")
                error_count += 1

        new_conn.commit()

        # Get total count in new database
        new_cursor.execute("SELECT COUNT(*) FROM emails")
        total_count = new_cursor.fetchone()[0]

        logging.info("Emails migration completed:")
        logging.info(f"- Successfully migrated: {success_count}")
        logging.info(f"- Skipped (duplicates): {skip_count}")
        logging.info(f"- Errors: {error_count}")
        logging.info(f"Total emails in new database: {total_count}")

    except sqlite3.Error as e:
        logging.error(f"Error during emails migration: {e}")
        raise

def migrate_benefits(old_conn, new_conn):
    """Migrate benefits from old database to new database."""
    old_cursor = old_conn.cursor()
    new_cursor = new_conn.cursor()
    
    # Check if benefits table exists in old database
    old_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='benefits'")
    if not old_cursor.fetchone():
        logger.warning("Benefits table not found in old database")
        return
    
    # Get all benefits from old database
    old_cursor.execute("SELECT id, url_id, benefit FROM benefits")
    benefits = old_cursor.fetchall()
    
    # Insert benefits into new database
    successful = 0
    skipped = 0
    errors = 0
    
    for benefit_record in benefits:
        try:
            benefit_id = benefit_record[0]
            url_id = benefit_record[1]
            description = benefit_record[2].strip() if benefit_record[2] else None
            
            if description:
                new_cursor.execute("""
                    INSERT INTO benefits (id, description, created_at, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (benefit_id, description))
                successful += 1
            else:
                skipped += 1
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                skipped += 1
                logger.debug(f"Skipped duplicate benefit: {description}")
            else:
                errors += 1
                logger.error(f"Error inserting benefit '{description}': {e}")
        except Exception as e:
            errors += 1
            logger.error(f"Unexpected error inserting benefit '{description}': {e}")
    
    new_conn.commit()
    logger.info(f"Benefits migration completed: {successful} successful, {skipped} skipped, {errors} errors")

def verify_benefits_migration(old_conn, new_conn):
    """Verify that benefits were migrated correctly."""
    old_cursor = old_conn.cursor()
    new_cursor = new_conn.cursor()
    
    # Check if benefits table exists in old database
    old_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='benefits'")
    if not old_cursor.fetchone():
        logger.warning("Benefits table not found in old database, skipping verification")
        return
    
    # Count benefits in old database
    old_cursor.execute("SELECT COUNT(*) FROM benefits")
    old_count = old_cursor.fetchone()[0]
    
    # Count benefits in new database
    new_cursor.execute("SELECT COUNT(*) FROM benefits")
    new_count = new_cursor.fetchone()[0]
    
    logger.info(f"Benefits migration verification: {old_count} in old database, {new_count} in new database")
    
    if old_count != new_count:
        logger.warning(f"Benefits count mismatch: {old_count} in old database, {new_count} in new database")
    else:
        logger.info("Benefits migration verified successfully")

def backup_database():
    """Create a backup of the existing database."""
    try:
        # Create backup directory if it doesn't exist
        backup_dir = "databases/backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{backup_dir}/recruitment_{timestamp}.db"
        
        # Copy the database file
        shutil.copy2(OLD_DB_PATH, backup_path)
        logger.info(f"Database backed up to {backup_path}")
    except Exception as e:
        logger.error(f"Error backing up database: {e}")
        raise

def verify_migration(old_conn, new_conn):
    """Verify the migration by comparing record counts between old and new databases."""
    logger.info("Starting migration verification...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Tables to verify
        tables = [
            'urls', 'companies', 'agencies', 'jobs', 'adverts',
            'skills', 'job_skills', 'qualifications', 'job_qualifications',
            'attributes', 'job_attributes', 'duties', 'job_duties',
            'locations', 'job_locations', 'phones', 'company_phones', 'agency_phones'
        ]
        
        for table in tables:
            try:
                # Get count from old database
                old_cursor.execute(f"SELECT COUNT(*) FROM {table}")
                old_count = old_cursor.fetchone()[0]
                
                # Get count from new database (using new_db schema)
                new_cursor.execute(f"SELECT COUNT(*) FROM new_db.{table}")
                new_count = new_cursor.fetchone()[0]
                
                logger.info(f"{table}: Old DB: {old_count}, New DB: {new_count}")
                
                if old_count != new_count:
                    logger.warning(f"Count mismatch for {table}: Old={old_count}, New={new_count}")
            except sqlite3.Error as e:
                logger.error(f"Error verifying {table}: {e}")
                continue
        
        logger.info("Migration verification completed")
        
    except Exception as e:
        logger.error(f"Error during migration verification: {e}")
        raise

def migrate_job_agencies(old_conn, new_conn):
    """Migrate job-agency relationships from old database to new database."""
    logger.info("Starting job-agency relationships migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get all job-agency relationships from old database
        old_cursor.execute("""
            SELECT DISTINCT ja.id as job_id, a.id as agency_id
            FROM job_adverts ja
            JOIN agency a ON ja.url_id = a.url_id
            WHERE ja.id IS NOT NULL AND a.id IS NOT NULL
        """)
        relationships = old_cursor.fetchall()
        
        logger.info(f"Found {len(relationships)} job-agency relationships to migrate")
        
        # Insert relationships into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for rel in relationships:
            try:
                new_cursor.execute("""
                    INSERT INTO job_agencies (job_id, agency_id)
                    VALUES (?, ?)
                """, (rel['job_id'], rel['agency_id']))
                successful += 1
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                    logger.debug(f"Skipped duplicate job-agency relationship: {rel}")
                else:
                    errors += 1
                    logger.error(f"Error inserting job-agency relationship: {rel}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting job-agency relationship: {rel}")

        new_conn.commit()
        logger.info(f"Job-agency relationships migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates): {skipped}")
        logger.info(f"- Errors: {errors}")
        
        return successful, skipped, errors
        
    except Exception as e:
        logger.error(f"Error during job-agency relationships migration: {e}")
        raise

def migrate_company_emails(old_conn, new_conn):
    """Migrate company-email relationships from old database to new database."""
    logger.info("Starting company-email relationships migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get all company-email relationships from old database
        old_cursor.execute("""
            SELECT DISTINCT c.id as company_id, e.id as email_id
            FROM company c
            JOIN email e ON c.url_id = e.url_id
            WHERE c.id IS NOT NULL AND e.id IS NOT NULL
        """)
        relationships = old_cursor.fetchall()
        
        logger.info(f"Found {len(relationships)} company-email relationships to migrate")
        
        # Insert relationships into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for rel in relationships:
            try:
                new_cursor.execute("""
                    INSERT INTO company_emails (company_id, email_id)
                    VALUES (?, ?)
                """, (rel['company_id'], rel['email_id']))
                successful += 1
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                    logger.debug(f"Skipped duplicate company-email relationship: {rel}")
                else:
                    errors += 1
                    logger.error(f"Error inserting company-email relationship: {rel}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting company-email relationship: {rel}")

        new_conn.commit()
        logger.info(f"Company-email relationships migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates): {skipped}")
        logger.info(f"- Errors: {errors}")
        
        return successful, skipped, errors
        
    except Exception as e:
        logger.error(f"Error during company-email relationships migration: {e}")
        raise

def migrate_agency_emails(old_conn, new_conn):
    """Migrate agency-email relationships from old database to new database."""
    logger.info("Starting agency-email relationships migration...")
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()
        
        # Get all agency-email relationships from old database
        old_cursor.execute("""
            SELECT DISTINCT a.id as agency_id, e.id as email_id
            FROM agency a
            JOIN email e ON a.url_id = e.url_id
            WHERE a.id IS NOT NULL AND e.id IS NOT NULL
        """)
        relationships = old_cursor.fetchall()
        
        logger.info(f"Found {len(relationships)} agency-email relationships to migrate")
        
        # Insert relationships into new database
        successful = 0
        skipped = 0
        errors = 0
        
        for rel in relationships:
            try:
                new_cursor.execute("""
                    INSERT INTO agency_emails (agency_id, email_id)
                    VALUES (?, ?)
                """, (rel['agency_id'], rel['email_id']))
                successful += 1
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                    logger.debug(f"Skipped duplicate agency-email relationship: {rel}")
                else:
                    errors += 1
                    logger.error(f"Error inserting agency-email relationship: {rel}")
            except Exception as e:
                errors += 1
                logger.error(f"Unexpected error inserting agency-email relationship: {rel}")

        new_conn.commit()
        logger.info(f"Agency-email relationships migration completed:")
        logger.info(f"- Successfully migrated: {successful}")
        logger.info(f"- Skipped (duplicates): {skipped}")
        logger.info(f"- Errors: {errors}")
        
        return successful, skipped, errors
        
    except Exception as e:
        logger.error(f"Error during agency-email relationships migration: {e}")
        raise

def main():
    """Main function to run the migration."""
    try:
        logger.info("Starting database migration process...")
        
        # Backup existing database
        logger.info("Backing up existing database...")
        backup_database()
        
        # Create new database schema
        logger.info("Creating new database schema...")
        with sqlite3.connect(OLD_DB_PATH) as old_conn:
            old_conn.row_factory = sqlite3.Row
            with sqlite3.connect(NEW_DB_PATH) as new_conn:
                new_conn.row_factory = sqlite3.Row
                
                # Create tables
                create_urls_table(new_conn)
                create_companies_table(new_conn)
                create_agencies_table(new_conn)
                create_jobs_table(new_conn)
                create_adverts_table(new_conn)
                create_skills_table(new_conn)
                create_job_skills_table(new_conn)
                create_qualifications_table(new_conn)
                create_job_qualifications_table(new_conn)
                create_attributes_table(new_conn)
                create_job_attributes_table(new_conn)
                create_duties_table(new_conn)
                create_job_duties_table(new_conn)
                create_locations_table(new_conn)
                create_job_locations_table(new_conn)
                create_phones_table(new_conn)
                create_phone_relationship_tables(new_conn)
                create_emails_table(new_conn)
                create_benefits_table(new_conn)
                
                # Migrate data
                migrate_agencies(old_conn, new_conn)
                migrate_urls(old_conn, new_conn)
                migrate_companies(old_conn, new_conn)
                migrate_jobs(old_conn, new_conn)
                migrate_adverts(old_conn, new_conn)
                migrate_skills(old_conn, new_conn)
                migrate_job_skills(old_conn, new_conn)
                migrate_qualifications(old_conn, new_conn)
                migrate_job_qualifications(old_conn, new_conn)
                migrate_attributes(old_conn, new_conn)
                migrate_job_attributes(old_conn, new_conn)
                migrate_duties(old_conn, new_conn)
                migrate_job_duties(old_conn, new_conn)
                migrate_locations(old_conn, new_conn)
                migrate_job_locations(old_conn, new_conn)
                migrate_phones(old_conn, new_conn)
                migrate_phone_relationships(old_conn, new_conn)
                migrate_emails(old_conn, new_conn)
                migrate_benefits(old_conn, new_conn)
                migrate_job_agencies(old_conn, new_conn)
                migrate_company_emails(old_conn, new_conn)
                migrate_agency_emails(old_conn, new_conn)
                
                # Verify migration
                verify_agencies_migration(old_conn, new_conn)
                verify_migration(old_conn, new_conn)
                verify_benefits_migration(old_conn, new_conn)
        
        logger.info("Migration process completed. Check migration.log for details.")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        # Close the database connections
        if 'old_conn' in locals():
            old_conn.close()
        if 'new_conn' in locals():
            new_conn.close()

if __name__ == "__main__":
    main() 