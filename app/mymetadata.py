import xml.etree.ElementTree as ET
from pathlib import Path
from pprint import pprint
import functools

import numpy as np
from xyconvert import wgs2gcj

import logging
logger = logging.getLogger(__name__)

import time
from typing import Callable, Any, Optional, Dict, Union, Tuple
import requests
import my_deco
from enum import Enum

class GeoAPIProvider(Enum):
    """Supported geocoding API providers"""
    AMAP = 'amap'          # Requires GCJ-02 coordinates
    LOCATIONIQ = 'locationiq'  # Uses WGS-84 coordinates
    # Add more providers as needed

class GeoProcessor:
    """Handles reverse geocoding with support for multiple APIs"""
    
    # API configurations - each with their specific parameter formats
    API_CONFIG = {
        GeoAPIProvider.AMAP: {
            'endpoint': "https://restapi.amap.com/v3/geocode/regeo",
            'key': '1861034270ab66c3be10f478330466fb',
            'params': {
                'output': 'json',
                'extensions': 'all',
                'radius': 1000,
            },
            'coord_format': '{lon},{lat}',  # AMap takes "lon,lat" string
            'requires_gcj02': True  # AMap needs Mars coordinates
        },
        GeoAPIProvider.LOCATIONIQ: {
            'endpoint': "https://us1.locationiq.com/v1/nearby",
            'key': 'pk.640a955650dce81e3442baa40151d0a6',
            'params': {
                'format': 'json',
                'accept-language': 'zh',
                'radius': 1000,
            },
            'coord_format': 'separate',  # LocationIQ takes separate lat/lon params
            'requires_gcj02': False  # Uses standard WGS-84
        }
    }
    
    @staticmethod
    def wgs84_to_gcj02(lon: float, lat: float) -> Tuple[float, float]:
        """
        Convert WGS-84 coordinates to GCJ-02 (Mars coordinates)
        
        Args:
            lon: Longitude in WGS-84
            lat: Latitude in WGS-84
            
        Returns:
            Tuple of (gcj_lon, gcj_lat) - Converted GCJ-02 coordinates
        """
        wgs_coords = np.array([[lon, lat]])
        gcj_coords = wgs2gcj(wgs_coords)
        return float(gcj_coords[0, 0]), float(gcj_coords[0, 1])
    
    @classmethod
    def reverse_geocode(
        cls,
        lon: float,
        lat: float,
        provider: GeoAPIProvider = GeoAPIProvider.LOCATIONIQ
    ) -> Dict:
        """
        Perform reverse geocoding using the specified provider
        
        Args:
            lon: Longitude in WGS-84
            lat: Latitude in WGS-84
            provider: Which geocoding API to use
            
        Returns:
            Dictionary with geocoding results
        """
        api_config = cls.API_CONFIG[provider]
        
        # Convert coordinates if required by the provider
        if api_config['requires_gcj02']:
            lon, lat = cls.wgs84_to_gcj02(lon, lat)
            conversion_metadata = {
                'original_coords': {'lon': lon, 'lat': lat, 'system': 'wgs84'},
                'converted_coords': {'lon': lon, 'lat': lat, 'system': 'gcj02'}
            }
        else:
            conversion_metadata = None
        
        # Make the API call
        result = cls._call_api(lon, lat, provider)
        
        # Add conversion metadata if applicable
        if conversion_metadata:
            result['conversion_metadata'] = conversion_metadata
            
        return result
    
    @classmethod
    @my_deco.retry_geo_api(max_retries=3, delay=1.0)
    def _call_api(cls, lon: float, lat: float, provider: GeoAPIProvider) -> Dict:
        """Internal method to call the specified geocoding API"""
        config = cls.API_CONFIG[provider]
        params = config['params'].copy()
        params['key'] = config['key']
        
        # Handle different coordinate parameter formats
        if config['coord_format'] == 'separate':
            params['lon'] = lon
            params['lat'] = lat
        else:
            params['location'] = config['coord_format'].format(lon=lon, lat=lat)
        
        try:
            response = requests.get(config['endpoint'], params=params)
            response.raise_for_status()
            data = response.json()

            # Provider-specific success checking
            if provider == GeoAPIProvider.AMAP and data.get('status') != '1':
                error_info = data.get('info', 'Unknown error')
                return {
                    'status': '0',
                    'info': error_info,
                    'infocode': data.get('infocode', '500')
                }
            elif provider == GeoAPIProvider.LOCATIONIQ and 'error' in data:
                return {
                    'status': '0',
                    'error': data.get('error', 'Unknown error')
                }
                
            return data
        except requests.exceptions.RequestException as e:
            return cls._format_error(provider, str(e))
        except Exception as e:
            return cls._format_error(provider, str(e))
    
    @classmethod
    def _format_error(cls, provider: GeoAPIProvider, message: str) -> Dict:
        """Format error response consistently across providers"""
        base_error = {
            'status': '0',
            'error': f"Request failed: {message}",
            'provider': provider.value
        }
        
        # Add provider-specific error fields
        if provider == GeoAPIProvider.AMAP:
            base_error['info'] = base_error.pop('error')
            base_error['infocode'] = '500'
        
        return base_error
    
    @classmethod
    def set_api_key(cls, provider: GeoAPIProvider, key: str):
        """Update API key for a specific provider"""
        cls.API_CONFIG[provider]['key'] = key

class PhotoMetadataExtractor:
    """Extracts and processes metadata from photo files and their associated XMP sidecars."""
    
    _namespaces = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'exif': 'http://ns.adobe.com/exif/1.0/',
        'tiff': 'http://ns.adobe.com/tiff/1.0/',
        'aux': 'http://ns.adobe.com/exif/1.0/aux/',
        'photoshop': 'http://ns.adobe.com/photoshop/1.0/',
        'dc': 'http://purl.org/dc/elements/1.1/',  # Added missing namespace
        'xmp': 'http://ns.adobe.com/xap/1.0/',
        'xmpMM': 'http://ns.adobe.com/xap/1.0/mm/'
    }
    
    @staticmethod
    def _dms_to_decimal(gps_str: str) -> Optional[float]:
        """Convert GPS coordinate string to decimal degrees."""
        if not gps_str:
            return None
        
        try:
            direction = gps_str[-1]
            parts = gps_str[:-1].replace(',', ' ').split()
            
            if len(parts) != 2:
                return None
            
            degrees = float(parts[0])
            minutes = float(parts[1])
            decimal = degrees + (minutes / 60)
            
            return -decimal if direction in ['S', 'W'] else decimal
        except (ValueError, AttributeError):
            return None
    
    @staticmethod
    def _find_xmp_file(image_path: Union[str, Path]) -> Optional[Path]:
        """Find associated XMP file for the given image path."""
        path = Path(image_path)
        
        # First try same filename with .xmp extension
        possible_xmp = path.with_suffix('.xmp')
        if possible_xmp.exists():
            return possible_xmp
        
        # Then try alternative pattern (filename.xmp for filename.ext)
        possible_xmp = path.parent / f"{path.stem}.xmp"
        return possible_xmp if possible_xmp.exists() else None
    
    @classmethod
    def _extract_from_xmp(cls, xmp_content: str) -> Dict[str, Dict[str, Union[str, float, int]]]:
        """Core metadata extraction from XMP content."""
        try:
            root = ET.fromstring(xmp_content)
            description = root.find('.//rdf:Description', {'rdf': cls._namespaces['rdf']})
            if description is None:
                return {}
            
            def get_attr(attr: str, namespace: str = '') -> str:
                """Helper to safely get attributes with namespace handling."""
                if namespace and namespace not in cls._namespaces:
                    return ''
                
                full_attr = f'{{{cls._namespaces[namespace]}}}{attr}' if namespace else attr
                return description.get(full_attr, '').strip()
            
            # GPS Data
            gps_lat = get_attr('GPSLatitude', 'exif')
            gps_lon = get_attr('GPSLongitude', 'exif')
            gps_lat_dec = cls._dms_to_decimal(gps_lat)
            gps_lon_dec = cls._dms_to_decimal(gps_lon)

            gps_resolved_d = {}
            if gps_lat_dec is not None and gps_lat_dec is not None:
                gps_resolved_d = GeoProcessor().reverse_geocode(gps_lon_dec, gps_lat_dec,
                                                                provider = GeoAPIProvider.LOCATIONIQ)
            
            # Process altitude
            altitude_str = get_attr('GPSAltitude', 'exif')
            altitude_meters = None
            if altitude_str and '/' in altitude_str:
                try:
                    numerator, denominator = map(float, altitude_str.split('/'))
                    altitude_meters = numerator / denominator
                except (ValueError, ZeroDivisionError):
                    pass
            
            # Camera Data
            focal_length = get_attr('FocalLength', 'exif')
            focal_length_mm = None
            if focal_length and '/' in focal_length:
                try:
                    numerator, denominator = map(float, focal_length.split('/'))
                    focal_length_mm = numerator / denominator
                except (ValueError, ZeroDivisionError):
                    pass
            
            aperture = get_attr('FNumber', 'exif')
            aperture_value = None
            if aperture and '/' in aperture:
                try:
                    numerator, denominator = map(float, aperture.split('/'))
                    aperture_value = numerator / denominator
                except (ValueError, ZeroDivisionError):
                    pass

            # Photo
            orientation = int(get_attr('Orientation', 'tiff') or 0)
            orientation_map = {
                1: 0,
                2: 0,
                3: 180,
                4: 180,
                5: 270,
                6: 270,
                7: 90,
                8: 90,
            }
            rotate = orientation_map[orientation]
            
            # Clean empty values from the results
            def clean_dict(d: dict) -> dict:
                return {k: v for k, v in d.items() if v not in [None, '', 0]}
            
            return {
                'gps': clean_dict({
                    'latitude': gps_lat,
                    'longitude': gps_lon,
                    'latitude_dec': gps_lat_dec,
                    'longitude_dec': gps_lon_dec,
                    'altitude': altitude_str,
                    'altitude_meters': altitude_meters,
                    'version': get_attr('GPSVersionID', 'exif')
                }),
                'gps_resolved': gps_resolved_d,
                'camera': clean_dict({
                    'make': get_attr('Make', 'tiff'),
                    'model': get_attr('Model', 'tiff'),
                    'serial': get_attr('SerialNumber', 'aux'),
                    'firmware': get_attr('Firmware', 'aux')
                }),
                'lens': clean_dict({
                    'model': get_attr('Lens', 'aux') or get_attr('LensModel', 'exif'),
                    'serial': get_attr('LensSerialNumber', 'aux'),
                    'focal_length': focal_length,
                    'focal_length_mm': focal_length_mm,
                    'aperture': aperture,
                    'aperture_value': aperture_value
                }),
                'photo': clean_dict({
                    'width': int(get_attr('ImageWidth', 'tiff') or 0),
                    'height': int(get_attr('ImageLength', 'tiff') or 0),
                    'create_date': get_attr('DateTimeOriginal', 'exif'),
                    'exposure': get_attr('ExposureTime', 'exif'),
                    'iso': int(get_attr('ISOSpeedRatings', 'exif') or 0),
                    'orientation': orientation,
                    'rotate': rotate,
                    'file_format': get_attr('format', 'dc')
                })
            }
        except ET.ParseError as e:
            print(f"Error parsing XMP: {e}")
            return {}
    
    @classmethod
    def extract(cls, file_path: Union[str, Path]) -> Dict[str, Dict[str, Union[str, float, int]]]:
        """
        Extract metadata from image file or its XMP sidecar.
        
        Args:
            file_path: Path to image file (CR3, JPG, etc.) or XMP file
            
        Returns:
            Dictionary containing categorized metadata
        """
        path = Path(file_path)
        
        # If input is already an XMP file
        if path.suffix.lower() == '.xmp':
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return cls._extract_from_xmp(f.read())
            except (IOError, OSError) as e:
                print(f"Error reading XMP file: {e}")
                return {}
        
        # For image files, find associated XMP
        xmp_path = cls._find_xmp_file(path)
        if xmp_path is None:
            print(f"No XMP sidecar found for {path}")
            return {}
        
        try:
            with open(xmp_path, 'r', encoding='utf-8') as f:
                return cls._extract_from_xmp(f.read())
        except (IOError, OSError) as e:
            print(f"Error reading XMP file: {e}")
            return {}



# Example usage
if __name__ == "__main__":
    
    # Can work with either:
    metadata = PhotoMetadataExtractor.extract("./dev_samples/test-sample-cr3/4H4A3790.xmp")
    pprint(metadata)
    
    metadata = PhotoMetadataExtractor.extract("./dev_samples/test-sample-cr3/4H4A3790.cr3")
    pprint(metadata)

    metadata = PhotoMetadataExtractor.extract("./dev_samples/test-sample-cr3/4H4A37901.cr3")
    pprint(metadata)
