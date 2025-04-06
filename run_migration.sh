#!/bin/bash

# Set up logging
LOG_FILE="migration.log"
exec 1> >(tee -a "$LOG_FILE") 2>&1

echo "Starting database migration process..."

# Create databases directory if it doesn't exist
mkdir -p databases

# Backup existing database
echo "Backing up existing database..."
if [ -f "databases/recruitment_new.db" ]; then
    cp databases/recruitment_new.db "databases/recruitment_new.db.$(date +%Y%m%d_%H%M%S).bak"
    # Drop the existing database
    rm databases/recruitment_new.db
fi

# Create new database and run schema creation script
echo "Creating new database schema..."
sqlite3 databases/recruitment_new.db < create_new_db.sql

# Run migration script
echo "Running migration script..."
python migrate_db.py

# Verify migration
echo "Verifying migration..."
python verify_migration.py

echo "Migration process completed. Check migration.log for details." 

chmod +x run_agencies_migration.sh 