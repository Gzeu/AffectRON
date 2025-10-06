-- Initialize database with Romanian financial data
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_extracted_data_created_at ON extracted_data(created_at);
CREATE INDEX IF NOT EXISTS idx_extracted_data_source_id ON extracted_data(source_id);
CREATE INDEX IF NOT EXISTS idx_sentiment_analysis_created_at ON sentiment_analysis(created_at);
CREATE INDEX IF NOT EXISTS idx_sentiment_analysis_data_id ON sentiment_analysis(data_id);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_market_data_currency_pair ON market_data(currency_pair);

-- Insert default data sources
INSERT INTO data_sources (name, source_type, url, is_active, extraction_interval) VALUES
('BNR Official', 'exchange_rates', 'https://www.bnr.ro/nbrfxrates.xml', true, 3600),
('Romanian Financial News', 'news', 'https://www.bursa.ro/rss', true, 1800),
('Twitter Financial', 'social_media', 'https://twitter.com/search', true, 900),
('ECB Rates', 'exchange_rates', 'https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml', true, 7200)
ON CONFLICT (name) DO NOTHING;
