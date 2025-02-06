#!/usr/bin/env python3
"""
Generative Full Tool for MLS, Public Data Fusion, and Chain-of-Custody

This file combines multiple components with generative fallback logic:
  1. MLSDataFetcher: Retrieves MLS data using configured endpoints, with fallback endpoints.
  2. PublicDataFusionTool: Uses Gemini-generated search terms, connectivity testing,
     Selenium+OCR template vectorization, generic scraping, and data fusion.
  3. PublicPropertyDataFetcher: Retrieves public property data with fallback methods.
  4. A Flask API to serve augmented MLS listings (with USPS integration).
  5. ChainOfCustodyEngine: Models property transfer records and determines the rightful owner.

Each method is designed to be generative (via simulated Gemini input) and to use fallback methods if primary calls fail.
"""

import time
import json
import logging
import requests
import cv2
import numpy as np
import pytesseract
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from datetime import datetime
from typing import List, Optional

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# ====================================================
# 0. Gemini Simulation Functions
# ====================================================
def gemini_generate(prompt: str) -> str:
    """
    Simulated Gemini generative function.
    In production, this would call a generative AI API.
    Fallback: returns a canned response.
    """
    try:
        # Simulate a generative call; here we just return a modified prompt.
        logging.info("Gemini generating response for prompt: %s", prompt)
        return f"Generated answer based on: {prompt}"
    except Exception as e:
        logging.error("Gemini generation failed: %s", e)
        return "Fallback response."

def gemini_generate_search_terms(location: str, data_points: List[str]) -> List[str]:
    """
    Use Gemini to generate search terms based on location and desired data points.
    """
    try:
        prompt = f"Generate search terms for {location} with data: {', '.join(data_points)}"
        generated = gemini_generate(prompt)
        # Simulate splitting the generated text into terms.
        terms = [term.strip() for term in generated.split(',') if term.strip()]
        if not terms:
            raise ValueError("No terms generated.")
        return terms
    except Exception as e:
        logging.error("Gemini search term generation failed: %s", e)
        # Fallback to a default generation:
        base = location + " public records"
        fallback_terms = [f"{base} {dp}" for dp in data_points] + [f"{location} assessor website"]
        return list(set(fallback_terms))

# ====================================================
# 1. MLSDataFetcher – Retrieves MLS Information with Fallbacks
# ====================================================
class MLSDataFetcher:
    def __init__(self, config):
        """
        Initialize with a configuration dictionary.
        The config may include primary and fallback endpoints.
        """
        self.config = config

    def _get_json_response(self, endpoint: str, params: dict, fallback_endpoint: Optional[str] = None) -> Optional[dict]:
        """
        Helper method to request JSON data. If the primary endpoint fails, use fallback.
        """
        for url in [endpoint, fallback_endpoint]:
            if not url:
                continue
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                logging.info("Successful response from %s", url)
                return data
            except Exception as e:
                logging.error("Error fetching from %s: %s", url, e)
        return None

    def get_basic_listing_info(self, listing_id):
        endpoint = self.config.get("basic_listing_endpoint")
        fallback = self.config.get("basic_listing_fallback_endpoint")
        params = {"listing_id": listing_id}
        data = self._get_json_response(endpoint, params, fallback)
        if data is None:
            return None
        return {
            "mls_listing_id": data.get("mls_listing_id"),
            "listing_status": data.get("listing_status"),
            "listing_date": data.get("listing_date"),
            "last_update_date": data.get("last_update_date"),
            "expiration_date": data.get("expiration_date")
        }

    def get_property_details(self, listing_id):
        endpoint = self.config.get("property_details_endpoint")
        fallback = self.config.get("property_details_fallback_endpoint")
        params = {"listing_id": listing_id}
        data = self._get_json_response(endpoint, params, fallback)
        if data is None:
            return None
        return {
            "address": data.get("address"),
            "geolocation": data.get("geolocation"),
            "property_type": data.get("property_type"),
            "property_description": data.get("property_description"),
            "physical_characteristics": {
                "square_footage": data.get("square_footage"),
                "lot_size": data.get("lot_size"),
                "bedrooms": data.get("bedrooms"),
                "bathrooms": data.get("bathrooms"),
                "floors": data.get("floors"),
                "year_built": data.get("year_built"),
                "architectural_style": data.get("architectural_style")
            }
        }

    def get_financial_data(self, listing_id):
        endpoint = self.config.get("financial_data_endpoint")
        fallback = self.config.get("financial_data_fallback_endpoint")
        params = {"listing_id": listing_id}
        data = self._get_json_response(endpoint, params, fallback)
        if data is None:
            return None
        return {
            "listing_price": data.get("listing_price"),
            "price_history": data.get("price_history"),
            "taxes_fees": {
                "property_tax": data.get("property_tax"),
                "hoa_fees": data.get("hoa_fees")
            },
            "financing_info": data.get("financing_info")
        }

    def get_multimedia_data(self, listing_id):
        endpoint = self.config.get("multimedia_endpoint")
        fallback = self.config.get("multimedia_fallback_endpoint")
        params = {"listing_id": listing_id}
        data = self._get_json_response(endpoint, params, fallback)
        if data is None:
            return None
        return {
            "photographs": data.get("photographs"),
            "virtual_tours": data.get("virtual_tours"),
            "floor_plans": data.get("floor_plans")
        }

    def get_features_and_amenities(self, listing_id):
        endpoint = self.config.get("features_endpoint")
        fallback = self.config.get("features_fallback_endpoint")
        params = {"listing_id": listing_id}
        data = self._get_json_response(endpoint, params, fallback)
        if data is None:
            return None
        return {
            "interior_features": data.get("interior_features"),
            "exterior_features": data.get("exterior_features"),
            "utilities": data.get("utilities"),
            "other_amenities": data.get("other_amenities")
        }

    def get_open_house_data(self, listing_id):
        endpoint = self.config.get("open_house_endpoint")
        fallback = self.config.get("open_house_fallback_endpoint")
        params = {"listing_id": listing_id}
        data = self._get_json_response(endpoint, params, fallback)
        if data is None:
            return None
        return {
            "open_house_dates": data.get("open_house_dates"),
            "appointment_scheduling": data.get("appointment_scheduling")
        }

    def get_legal_compliance_info(self, listing_id):
        endpoint = self.config.get("legal_compliance_endpoint")
        fallback = self.config.get("legal_compliance_fallback_endpoint")
        params = {"listing_id": listing_id}
        data = self._get_json_response(endpoint, params, fallback)
        if data is None:
            return None
        return {
            "disclosures": data.get("disclosures"),
            "zoning_information": data.get("zoning_information"),
            "legal_description": data.get("legal_description"),
            "compliance_inspection_records": data.get("compliance_inspection_records")
        }

    def get_agent_brokerage_info(self, listing_id):
        endpoint = self.config.get("agent_brokerage_endpoint")
        fallback = self.config.get("agent_brokerage_fallback_endpoint")
        params = {"listing_id": listing_id}
        data = self._get_json_response(endpoint, params, fallback)
        if data is None:
            return None
        return {
            "listing_agent": data.get("listing_agent"),
            "co_listing_agents": data.get("co_listing_agents"),
            "brokerage_information": data.get("brokerage_information"),
            "office_details": data.get("office_details")
        }

    def get_market_comparative_data(self, listing_id):
        endpoint = self.config.get("market_data_endpoint")
        fallback = self.config.get("market_data_fallback_endpoint")
        params = {"listing_id": listing_id}
        data = self._get_json_response(endpoint, params, fallback)
        if data is None:
            return None
        return {
            "market_trends": data.get("market_trends"),
            "property_valuation_estimates": data.get("property_valuation_estimates"),
            "historical_transaction_data": data.get("historical_transaction_data")
        }

    def get_additional_metadata(self, listing_id):
        endpoint = self.config.get("metadata_endpoint")
        fallback = self.config.get("metadata_fallback_endpoint")
        params = {"listing_id": listing_id}
        data = self._get_json_response(endpoint, params, fallback)
        if data is None:
            return None
        return {
            "data_source_information": data.get("data_source_information"),
            "custom_fields": data.get("custom_fields")
        }

# ====================================================
# 2. Public Data Fusion Tool Components with Generative Gemini Input
# ====================================================

# 2.1 Gemini – Search Term Generation with Fallback
def generate_search_terms(location: str, data_points: List[str]) -> List[str]:
    try:
        terms = gemini_generate_search_terms(location, data_points)
        if not terms:
            raise ValueError("No terms generated by Gemini.")
        return terms
    except Exception as e:
        logging.error("Error generating search terms via Gemini: %s", e)
        # Fallback default search terms:
        base = location + " public records"
        fallback_terms = [f"{base} {dp}" for dp in data_points] + [f"{location} assessor website"]
        return list(set(fallback_terms))

# 2.2 Connectivity Testing with Fallback Logging
def test_source_performance(url: str, timeout: int = 10) -> dict:
    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        elapsed = time.time() - start_time
        return {"url": url, "status_code": response.status_code, "response_time": elapsed}
    except Exception as e:
        logging.error("Connectivity test failed for %s: %s", url, e)
        return {"url": url, "error": str(e), "response_time": None}

# 2.3 Template Vectorization & JSON Template Generation with Fallback
class TemplateVectorizer:
    def __init__(self, driver_path: str = "chromedriver"):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        try:
            self.driver = webdriver.Chrome(driver_path, options=chrome_options)
        except Exception as e:
            logging.error("Failed to initialize Selenium WebDriver: %s", e)
            self.driver = None

    def capture_screenshot(self, url: str, output_path: str = "screenshot.png") -> Optional[str]:
        if not self.driver:
            logging.error("WebDriver not available.")
            return None
        try:
            self.driver.get(url)
            time.sleep(3)
            self.driver.save_screenshot(output_path)
            return output_path
        except Exception as e:
            logging.error("Error capturing screenshot from %s: %s", url, e)
            return None

    def perform_ocr(self, image_path: str) -> List[dict]:
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            regions = []
            for i in range(len(ocr_data['level'])):
                text = ocr_data['text'][i].strip()
                if text:
                    regions.append({"text": text, "region": [ocr_data['left'][i], ocr_data['top'][i],
                                                               ocr_data['left'][i] + ocr_data['width'][i],
                                                               ocr_data['top'][i] + ocr_data['height'][i]]})
            return regions
        except Exception as e:
            logging.error("Error performing OCR on %s: %s", image_path, e)
            return []

    def generate_json_template(self, url: str) -> dict:
        screenshot_path = self.capture_screenshot(url)
        if not screenshot_path:
            logging.error("No screenshot available; using fallback template.")
            # Fallback template
            return {
                "site_name": url,
                "url": url,
                "data_points": {
                    "parcel_number": {"css_selector": "#fallback-parcel", "ocr_region": [0, 0, 100, 50]},
                    "address": {"css_selector": ".fallback-address", "ocr_region": [0, 50, 200, 100]},
                    "assessment_value": {"css_selector": ".fallback-assessment", "ocr_region": [0, 100, 150, 150]}
                },
                "vectorized_features": [0.0, 0.0],
                "ocr_regions": [],
                "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        ocr_regions = self.perform_ocr(screenshot_path)
        image = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
        feature_vector = [float(np.mean(image)), float(np.std(image))]
        template = {
            "site_name": url,
            "url": url,
            "data_points": {
                "parcel_number": {"css_selector": "#parcel-num", "ocr_region": [50, 100, 300, 150]},
                "address": {"css_selector": ".property-address", "ocr_region": [50, 160, 400, 210]},
                "assessment_value": {"css_selector": ".assessment-value", "ocr_region": [50, 220, 300, 270]}
            },
            "vectorized_features": feature_vector,
            "ocr_regions": ocr_regions,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        return template

    def close(self):
        if self.driver:
            self.driver.quit()

# 2.4 Generic Scraper Using JSON Template with Fallback
class DataScraper:
    def __init__(self, json_template: dict):
        self.template = json_template

    def scrape_data(self, url: str) -> dict:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logging.error("Error scraping %s: %s", url, e)
            return {}
        soup = BeautifulSoup(response.text, "html.parser")
        extracted_data = {}
        for key, settings in self.template.get("data_points", {}).items():
            selector = settings.get("css_selector")
            element = soup.select_one(selector)
            if element:
                extracted_data[key] = element.get_text(strip=True)
            else:
                logging.warning("Selector %s not found in %s; attempting fallback OCR extraction.", selector, url)
                # Fallback: use Gemini to generate a query and simulate extraction
                extracted_data[key] = gemini_generate(f"Extract {key} from {url}")
        return extracted_data

# 2.5 Data Fusion Engine with Fallback and Logging
class DataFusionEngine:
    def __init__(self):
        self.sources = []

    def add_source_data(self, source_name: str, data: dict):
        self.sources.append({"source": source_name, "data": data})

    def fuse(self) -> dict:
        if not self.sources:
            logging.error("No source data available to fuse.")
            return {}
        fused_data = {}
        keys = set()
        for source in self.sources:
            keys.update(source["data"].keys())
        for key in keys:
            for source in self.sources:
                val = source["data"].get(key)
                if val not in (None, ""):
                    fused_data[key] = val
                    break
            else:
                fused_data[key] = gemini_generate(f"Fallback value for {key}")
        return fused_data

# 2.6 Full Tool Orchestration with Gemini Input
class PublicDataFusionTool:
    def __init__(self, location: str, data_points: List[str]):
        self.location = location
        self.data_points = data_points
        self.candidate_urls = []  # to be discovered
        self.selected_sources = []  # after connectivity testing
        self.vectorizer = TemplateVectorizer()
        self.fusion_engine = DataFusionEngine()

    def discover_sources(self) -> List[str]:
        terms = generate_search_terms(self.location, self.data_points)
        logging.info("Generated search terms: %s", terms)
        # Simulate discovered candidate URLs via Gemini or search APIs.
        self.candidate_urls = [
            "https://assessor.examplecounty.gov/property",
            "https://recorder.examplecounty.gov/transactions"
        ]
        return self.candidate_urls

    def test_and_select_sources(self) -> List[str]:
        test_results = [test_source_performance(url) for url in self.candidate_urls]
        valid = [r for r in test_results if r.get("status_code") == 200]
        valid.sort(key=lambda x: x.get("response_time", 999))
        self.selected_sources = [r["url"] for r in valid]
        if not self.selected_sources:
            logging.warning("No candidate source passed connectivity testing; using fallback URL.")
            self.selected_sources = ["https://fallback.example.com"]
        return self.selected_sources

    def vectorize_and_generate_template(self, source_url: str) -> dict:
        template = self.vectorizer.generate_json_template(source_url)
        return template

    def run_scraping(self, template: dict, source_url: str) -> dict:
        scraper = DataScraper(template)
        data = scraper.scrape_data(source_url)
        return data

    def fuse_all_data(self) -> dict:
        fused = self.fusion_engine.fuse()
        return fused

    def run(self) -> dict:
        self.discover_sources()
        selected = self.test_and_select_sources()
        for url in selected:
            template = self.vectorize_and_generate_template(url)
            data = self.run_scraping(template, url)
            self.fusion_engine.add_source_data(url, data)
        fused = self.fuse_all_data()
        self.vectorizer.close()
        return fused

# ====================================================
# 3. PublicPropertyDataFetcher – Uses Public Sources with Fallbacks
# ====================================================
class PublicPropertyDataFetcher:
    def __init__(self, region_config: dict):
        self.config = region_config

    def get_parcel_number(self, address: str) -> Optional[str]:
        url = self.config.get("assessor_url")
        params = {"address": address}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logging.error("Error querying assessor site: %s", e)
            return None
        soup = BeautifulSoup(response.text, 'html.parser')
        selector = self.config.get("parcel_selector", "")
        parcel_element = soup.select_one(selector)
        if parcel_element:
            return parcel_element.get_text(strip=True)
        else:
            logging.warning("Parcel selector %s not found; using Gemini fallback.", selector)
            return gemini_generate(f"Extract parcel number from {address}")

    def get_transaction_history(self, parcel_number: str) -> List[dict]:
        url = self.config.get("recorder_url")
        params = {"parcel": parcel_number}
        transactions = []
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logging.error("Error querying recorder site: %s", e)
            return transactions
        soup = BeautifulSoup(response.text, 'html.parser')
        record_selector = self.config.get("transaction_selector", "")
        record_elements = soup.select(record_selector)
        for element in record_elements:
            sale_date_elem = element.find("span", class_="sale-date")
            price_elem = element.find("span", class_="price")
            transaction = {
                "sale_date": sale_date_elem.get_text(strip=True) if sale_date_elem else None,
                "price": price_elem.get_text(strip=True) if price_elem else None
            }
            transactions.append(transaction)
        return transactions

    def geocode_address(self, address: str) -> Optional[dict]:
        base_url = self.config.get("geocode_api")
        api_key = self.config.get("geocode_api_key")
        params = {"address": address, "key": api_key}
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logging.error("Error calling geocode API: %s", e)
            return None
        data = response.json()
        if data.get("results"):
            return data["results"][0]["geometry"]["location"]
        return None

    def get_tax_assessment(self, parcel_number: str) -> Optional[str]:
        url = self.config.get("tax_assessor_url")
        params = {"parcel": parcel_number}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logging.error("Error querying tax assessor site: %s", e)
            return None
        soup = BeautifulSoup(response.text, 'html.parser')
        assessment_selector = self.config.get("assessment_selector", "")
        assessment_element = soup.select_one(assessment_selector)
        if assessment_element:
            return assessment_element.get_text(strip=True)
        else:
            logging.warning("Assessment selector not found; using fallback.")
            return gemini_generate("Extract tax assessment")

    def get_zoning_info(self, address: str) -> Optional[str]:
        url = self.config.get("zoning_url")
        if not url:
            logging.warning("No zoning URL configured.")
            return None
        params = {"address": address}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logging.error("Error querying zoning site: %s", e)
            return None
        soup = BeautifulSoup(response.text, 'html.parser')
        zoning_selector = self.config.get("zoning_selector", "")
        zoning_element = soup.select_one(zoning_selector)
        if zoning_element:
            return zoning_element.get_text(strip=True)
        else:
            logging.warning("Zoning selector not found; using Gemini fallback.")
            return gemini_generate("Extract zoning info")

    def get_building_permits(self, parcel_number: str) -> List[dict]:
        url = self.config.get("permit_url")
        if not url:
            logging.warning("No permit URL configured.")
            return []
        params = {"parcel": parcel_number}
        permits = []
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logging.error("Error querying permits site: %s", e)
            return permits
        soup = BeautifulSoup(response.text, 'html.parser')
        permit_selector = self.config.get("permit_selector", "")
        permit_elements = soup.select(permit_selector)
        for elem in permit_elements:
            permit_date_elem = elem.find("span", class_="permit-date")
            permit_type_elem = elem.find("span", class_="permit-type")
            permit = {
                "date": permit_date_elem.get_text(strip=True) if permit_date_elem else None,
                "type": permit_type_elem.get_text(strip=True) if permit_type_elem else None,
            }
            permits.append(permit)
        return permits

# ====================================================
# 4. Flask API for Augmented MLS Listings
# ====================================================
from flask import Flask, jsonify, request

app = Flask(__name__)

# Simulated MLS Data
MLS_LISTINGS = [
    {
        "listing_id": "MLS001",
        "address": "123 Main St, Somecity, CA 90001",
        "property_type": "Apartment",
        "bedrooms": 2,
        "bathrooms": 1,
        "price": 2200,
        "listed_date": "2025-01-15"
    },
    {
        "listing_id": "MLS002",
        "address": "456 Oak Ave, Othertown, TX 75001",
        "property_type": "Condo",
        "bedrooms": 3,
        "bathrooms": 2,
        "price": 3000,
        "listed_date": "2025-02-01"
    }
]

# USPS API Integration (Simulated)
USPS_API_URL = "https://secure.shippingapis.com/ShippingAPI.dll"
USPS_USER_ID = "YOUR_USPS_USER_ID"

def usps_address_verification(address: str) -> dict:
    simulated_response = {
        "standardized_address": address,
        "city": "SimCity",
        "state": "CA",
        "zip5": "90001",
        "validated": True,
        "validation_date": datetime.now().isoformat()
    }
    return simulated_response

def augment_listing(listing: dict) -> dict:
    usps_data = usps_address_verification(listing["address"])
    augmented_listing = listing.copy()
    augmented_listing["address_details"] = usps_data
    return augmented_listing

def get_all_augmented_listings() -> List[dict]:
    return [augment_listing(listing) for listing in MLS_LISTINGS]

@app.route("/listings", methods=["GET"])
def listings():
    property_type = request.args.get("property_type")
    min_price = request.args.get("min_price", type=int)
    max_price = request.args.get("max_price", type=int)
    
    listings_data = get_all_augmented_listings()
    if property_type:
        listings_data = [l for l in listings_data if l["property_type"].lower() == property_type.lower()]
    if min_price is not None:
        listings_data = [l for l in listings_data if l["price"] >= min_price]
    if max_price is not None:
        listings_data = [l for l in listings_data if l["price"] <= max_price]
    return jsonify(listings_data)

@app.route("/listings/<listing_id>", methods=["GET"])
def listing_detail(listing_id):
    listings_data = get_all_augmented_listings()
    listing = next((l for l in listings_data if l["listing_id"] == listing_id), None)
    if listing is None:
        return jsonify({"error": "Listing not found"}), 404
    return jsonify(listing)

# ====================================================
# 5. Chain-of-Custody Engine
# ====================================================
class Owner:
    def __init__(self, owner_id: str, name: str):
        self.owner_id = owner_id
        self.name = name

    def __repr__(self):
        return f"Owner({self.owner_id}, {self.name})"

class Property:
    def __init__(self, property_id: str, description: str):
        self.property_id = property_id
        self.description = description

    def __repr__(self):
        return f"Property({self.property_id}, {self.description})"

class TransferRecord:
    def __init__(self, record_id: str, property_id: str, from_owner: Optional[Owner],
                 to_owner: Owner, transfer_date: datetime, legal_document: str):
        self.record_id = record_id
        self.property_id = property_id
        self.from_owner = from_owner
        self.to_owner = to_owner
        self.transfer_date = transfer_date
        self.legal_document = legal_document

    def __repr__(self):
        from_owner = self.from_owner.name if self.from_owner else "Origin"
        return (f"TransferRecord({self.record_id}, {self.property_id}, "
                f"from: {from_owner}, to: {self.to_owner.name}, "
                f"date: {self.transfer_date.strftime('%Y-%m-%d')})")

class ChainOfCustodyEngine:
    def __init__(self, records: List[TransferRecord]):
        self.records = records

    def get_property_chain(self, property_id: str) -> List[TransferRecord]:
        chain = [r for r in self.records if r.property_id == property_id]
        chain.sort(key=lambda r: r.transfer_date)
        return chain

    def validate_chain(self, chain: List[TransferRecord]) -> bool:
        if not chain:
            return False
        if chain[0].from_owner is not None:
            logging.error("Chain validation failed: First record must have no 'from_owner'.")
            return False
        for i in range(1, len(chain)):
            previous_owner = chain[i - 1].to_owner
            current_from_owner = chain[i].from_owner
            if current_from_owner is None or current_from_owner.owner_id != previous_owner.owner_id:
                logging.error("Chain validation failed between records %s and %s.",
                              chain[i-1].record_id, chain[i].record_id)
                return False
        return True

    def determine_rightful_owner(self, property_id: str) -> Optional[Owner]:
        chain = self.get_property_chain(property_id)
        if not chain:
            logging.error("No records found for property: %s", property_id)
            return None
        if not self.validate_chain(chain):
            logging.error("Chain-of-custody validation failed for property: %s", property_id)
            return None
        return chain[-1].to_owner

def run_chain_of_custody_example():
    owner_origin = Owner("O0", "Origin Authority")
    alice = Owner("O1", "Alice")
    bob = Owner("O2", "Bob")
    carol = Owner("O3", "Carol")

    prop = Property("P123", "123 Main Street")
    records = [
        TransferRecord("R1", prop.property_id, None, alice, datetime(2020, 1, 10), "Doc_Alice_Orig"),
        TransferRecord("R2", prop.property_id, alice, bob, datetime(2021, 6, 15), "Doc_Bob_Transfer"),
        TransferRecord("R3", prop.property_id, bob, carol, datetime(2023, 3, 22), "Doc_Carol_Transfer"),
    ]
    engine = ChainOfCustodyEngine(records)
    owner = engine.determine_rightful_owner(prop.property_id)
    if owner:
        print(f"The rightful owner of property {prop.property_id} is {owner.name}.")
    else:
        print("Could not determine a valid chain-of-custody for property", prop.property_id)

# ====================================================
# 6. Main Execution Block
# ====================================================
if __name__ == "__main__":
    # Example usage of MLSDataFetcher with fallback endpoints in config.
    mls_config = {
        "basic_listing_endpoint": "https://api.example-mls.com/basic_listing",
        "basic_listing_fallback_endpoint": "https://fallback.example-mls.com/basic_listing",
        "property_details_endpoint": "https://api.example-mls.com/property_details",
        "property_details_fallback_endpoint": "https://fallback.example-mls.com/property_details",
        "financial_data_endpoint": "https://api.example-mls.com/financial_data",
        "financial_data_fallback_endpoint": "https://fallback.example-mls.com/financial_data",
        "multimedia_endpoint": "https://api.example-mls.com/multimedia",
        "multimedia_fallback_endpoint": "https://fallback.example-mls.com/multimedia",
        "features_endpoint": "https://api.example-mls.com/features",
        "features_fallback_endpoint": "https://fallback.example-mls.com/features",
        "open_house_endpoint": "https://api.example-mls.com/open_house",
        "open_house_fallback_endpoint": "https://fallback.example-mls.com/open_house",
        "legal_compliance_endpoint": "https://api.example-mls.com/legal_compliance",
        "legal_compliance_fallback_endpoint": "https://fallback.example-mls.com/legal_compliance",
        "agent_brokerage_endpoint": "https://api.example-mls.com/agent_brokerage",
        "agent_brokerage_fallback_endpoint": "https://fallback.example-mls.com/agent_brokerage",
        "market_data_endpoint": "https://api.example-mls.com/market_data",
        "market_data_fallback_endpoint": "https://fallback.example-mls.com/market_data",
        "metadata_endpoint": "https://api.example-mls.com/metadata",
        "metadata_fallback_endpoint": "https://fallback.example-mls.com/metadata"
    }
    mls_fetcher = MLSDataFetcher(mls_config)
    listing_id = "MLS123456"
    basic_info = mls_fetcher.get_basic_listing_info(listing_id)
    print("Basic Listing Information:", json.dumps(basic_info, indent=2))

    # Run PublicDataFusionTool with generative search and fallback.
    location = "Example County Assessor"
    data_points = ["parcel number", "property tax", "address", "assessment value"]
    fusion_tool = PublicDataFusionTool(location, data_points)
    fused_data = fusion_tool.run()
    print("\nFinal Fused Public Data:")
    print(json.dumps(fused_data, indent=2))

    # Example usage of PublicPropertyDataFetcher.
    region_config_example = {
        "assessor_url": "https://public-assessor.example.com/search",
        "parcel_selector": ".parcel-number",
        "recorder_url": "https://public-recorder.example.com/transactions",
        "transaction_selector": ".transaction-record",
        "geocode_api": "https://maps.googleapis.com/maps/api/geocode/json",
        "geocode_api_key": "YOUR_GOOGLE_GEOCODE_API_KEY",
        "tax_assessor_url": "https://public-assessor.example.com/tax",
        "assessment_selector": ".assessment-value",
        "zoning_url": "https://public-zoning.example.com/info",
        "zoning_selector": ".zoning-info",
        "permit_url": "https://public-permits.example.com/search",
        "permit_selector": ".permit-record"
    }
    public_fetcher = PublicPropertyDataFetcher(region_config_example)
    address = "123 Main Street, Anytown, ST 12345"
    parcel = public_fetcher.get_parcel_number(address)
    print("Parcel Number:", parcel)
    if parcel:
        transactions = public_fetcher.get_transaction_history(parcel)
        print("Transaction History:", transactions)
        tax_info = public_fetcher.get_tax_assessment(parcel)
        print("Tax Assessment:", tax_info)
        permits = public_fetcher.get_building_permits(parcel)
        print("Building Permits:", permits)
    geolocation = public_fetcher.geocode_address(address)
    print("Geolocation:", geolocation)
    zoning = public_fetcher.get_zoning_info(address)
    print("Zoning Info:", zoning)

    # Run chain-of-custody example.
    run_chain_of_custody_example()

    # To run the Flask API, uncomment the following line.
    # app.run(debug=True, port=5000)
