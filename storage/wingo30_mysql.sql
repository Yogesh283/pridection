-- XAMPP / phpMyAdmin helper (optional).
-- App also auto-creates this database and tables on first run.

CREATE DATABASE IF NOT EXISTS wingo30
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE wingo30;

CREATE TABLE IF NOT EXISTS rounds (
    id INT NOT NULL AUTO_INCREMENT,
    period VARCHAR(64) NOT NULL,
    number INT NOT NULL,
    color VARCHAR(64) NULL,
    timestamp VARCHAR(64) NULL,
    raw_json LONGTEXT NULL,
    created_at VARCHAR(64) NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_rounds_period (period)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS predictions (
    id INT NOT NULL AUTO_INCREMENT,
    target_period VARCHAR(64) NOT NULL,
    predicted_number INT NULL,
    predicted_color VARCHAR(64) NULL,
    number_probability DOUBLE NULL,
    color_probability DOUBLE NULL,
    model_name VARCHAR(64) NULL,
    created_at VARCHAR(64) NULL,
    actual_number INT NULL,
    actual_color VARCHAR(64) NULL,
    number_correct TINYINT NULL,
    color_correct TINYINT NULL,
    resolved_at VARCHAR(64) NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS model_metrics (
    id INT NOT NULL AUTO_INCREMENT,
    model_name VARCHAR(64) NOT NULL,
    sample_size INT NULL,
    number_accuracy DOUBLE NULL,
    color_accuracy DOUBLE NULL,
    updated_at VARCHAR(64) NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_model_metrics_name (model_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
