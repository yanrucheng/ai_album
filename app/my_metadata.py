import xml.etree.ElementTree as ET
from pathlib import Path
from pprint import pprint
import functools

import numpy as np
from xyconvert import wgs2gcj, gcj2wgs

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

class MapDatum(Enum):
    GCJ02 = 'gcj02'
    WGS84 = 'wgs84'

class GeoProcessor:
    """Handles reverse geocoding with support for multiple APIs, providing standardized output."""
    
    # Assume these are defined elsewhere as needed
    AMAP_API_KEY = '1861034270ab66c3be10f478330466fb'
    LOCATIONIQ_API_KEY = 'pk.640a955650dce81e3442baa40151d0a6'
    _last_used_provider = GeoAPIProvider.LOCATIONIQ  # Default to LocationIQ

    @classmethod
    def convert(cls, lon, lat, from_datum, to_datum):
        if from_datum == to_datum:
            return lon, lat
        if from_datum == MapDatum.GCJ02 and to_datum == MapDatum.WGS84:
            return cls.gcj02_to_wgs84(lon, lat)
        if from_datum == MapDatum.WGS84 and to_datum == MapDatum.GCJ02:
            return cls.wgs84_to_gcj02(lon, lat)
        raise NotImplementedError()

    @staticmethod
    def gcj02_to_wgs84(lon: float, lat: float) -> Tuple[float, float]:
        gcj_coords = np.array([[lon, lat]])
        wgs_coords = gcj2wgs(gcj_coords)
        return float(wgs_coords[0, 0]), float(wgs_coords[0, 1])

    @staticmethod
    def wgs84_to_gcj02(lon: float, lat: float) -> Tuple[float, float]:
        wgs_coords = np.array([[lon, lat]])
        gcj_coords = wgs2gcj(wgs_coords)
        return float(gcj_coords[0, 0]), float(gcj_coords[0, 1])

    @classmethod
    def reverse_geocode(
        cls,
        lon: float,
        lat: float,
        datum: MapDatum = MapDatum.WGS84,
    ) -> Dict:
        """
        External method for reverse geocoding that manages provider switching.
        
        Args:
            lon: Longitude in WGS-84
            lat: Latitude in WGS-84
            datum: Coordinate system datum (default: WGS84)
            
        Returns:
            Dictionary with geocoding results
        """
        # First try with the last used provider
        provider = cls._last_used_provider
        result = cls._reverse_geocode(lon, lat, provider, datum)
        
        # Check if we need to switch providers
        if provider == GeoAPIProvider.LOCATIONIQ and cls._is_in_china(result):
            # Switch to AMap and try again
            new_result = cls._reverse_geocode(lon, lat, GeoAPIProvider.AMAP, datum)
            if new_result.get('status') == 'success' and new_result.get('location'):
                cls._last_used_provider = GeoAPIProvider.AMAP
                return new_result
        elif provider == GeoAPIProvider.AMAP and cls._is_empty(result):
            # Switch to LocationIQ and try again
            new_result = cls._reverse_geocode(lon, lat, GeoAPIProvider.LOCATIONIQ, datum)
            if new_result.get('status') == 'success' and new_result.get('location'):
                cls._last_used_provider = GeoAPIProvider.LOCATIONIQ
                return new_result
        
        # Return the original result if no switch was needed or if the switch failed
        return result

    @classmethod
    def _reverse_geocode(
        cls,
        lon: float,
        lat: float,
        provider: GeoAPIProvider,
        datum: MapDatum,
    ) -> Dict:
        """
        Internal method for reverse geocoding with a specified provider and datum.
        
        Args:
            lon: Longitude
            lat: Latitude
            provider: Geocoding API provider
            datum: Coordinate system datum
            
        Returns:
            Dictionary with geocoding results
        """
        # Determine target datum for the provider
        if provider == GeoAPIProvider.AMAP:
            to_datum = MapDatum.GCJ02
        elif provider == GeoAPIProvider.LOCATIONIQ:
            to_datum = MapDatum.WGS84
        else:
            raise NotImplementedError(f"Provider {provider} not supported")

        # Convert coordinates
        lon_conv, lat_conv = cls.convert(lon, lat, from_datum=datum, to_datum=to_datum)
        conversion_metadata = {
            'original_coords': {'lon': lon, 'lat': lat, 'datum': datum.value},
            'converted_coords': {'lon': lon_conv, 'lat': lat_conv, 'datum': to_datum.value}
        }

        # Process based on provider
        if provider == GeoAPIProvider.AMAP:
            result = cls._process_amap(lon_conv, lat_conv)
        elif provider == GeoAPIProvider.LOCATIONIQ:
            result = cls._process_locationiq(lon_conv, lat_conv)
        else:
            raise NotImplementedError(f"Processing for {provider} not implemented")

        result['conversion_metadata'] = conversion_metadata
        result['provider'] = provider.value
        return result

    @classmethod
    def _is_in_china(cls, result: Dict) -> bool:
        """Check if the result is in China based on the address components."""
        if result.get('status') != 'success':
            return False
        address = result.get('location', {}).get('components', {})
        country = address.get('country', '').lower()
        return 'china' in country or '中国' in country

    @classmethod
    def _is_empty(cls, result: Dict) -> bool:
        """Check if the result is empty or invalid."""
        if result.get('status') != 'success':
            return True
        location = result.get('location')
        return not location or not location.get('formatted_address')

    @classmethod
    def _process_amap(cls, lon: float, lat: float) -> Dict:
        params = {
            'key': cls.AMAP_API_KEY,
            'output': 'json',
            'extensions': 'all',
            'radius': 1000,
            'location': f"{lon:.6f},{lat:.6f}",
            'accept-language': 'zh',
        }
        response = cls._call_amap_api(params)
        if response.get('status') == '1':
            regeocode = response.get('regeocode', {})
            location = {
                'formatted_address': regeocode.get('formatted_address'),
                'components': regeocode.get('addressComponent', {})
            }
            pois = [cls._standardize_amap_poi(poi) for poi in regeocode.get('pois', [])]
            return {
                'status': 'success',
                'location': location,
                'pois': pois,
                'error': None
            }
        else:
            return {
                'status': 'error',
                'error': response.get('info', 'AMAP API error'),
                'location': None,
                'pois': []
            }

    @classmethod
    @my_deco.retry_geo_api(max_retries=3, delay=1.0)
    def _call_amap_api(cls, params: Dict) -> Dict:
        endpoint = "https://restapi.amap.com/v3/geocode/regeo"
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get('status') == '1':
                return data
            else:
                return {
                    'status': '0',
                    'info': data.get('info', 'Unknown error'),
                    'infocode': data.get('infocode', '500')
                }
        except requests.exceptions.RequestException as e:
            return cls._format_error(GeoAPIProvider.AMAP, str(e))
        except Exception as e:
            return cls._format_error(GeoAPIProvider.AMAP, str(e))

    @classmethod
    def _process_locationiq(cls, lon: float, lat: float) -> Dict:
        # Get location data
        reverse_params = {
            'key': cls.LOCATIONIQ_API_KEY,
            'format': 'json',
            'lat': lat,
            'lon': lon,
            'accept-language': 'zh',
        }
        reverse_response = cls._call_locationiq_reverse(reverse_params)

        # Get POIs
        nearby_params = {
            'key': cls.LOCATIONIQ_API_KEY,
            'format': 'json',
            'lat': lat,
            'lon': lon,
            'radius': 1000,
            'accept-language': 'zh',
        }
        nearby_response = cls._call_locationiq_nearby(nearby_params)

        # Parse location
        location = None
        if not reverse_response.get('error'):
            location = {
                'formatted_address': reverse_response.get('display_name'),
                'components': reverse_response.get('address', {})
            }

        # Parse POIs
        pois = []
        if isinstance(nearby_response, list):
            pois = [cls._standardize_locationiq_poi(poi) for poi in nearby_response]
        elif isinstance(nearby_response, dict) and nearby_response.get('error'):
            pass  # Handle error if needed

        # Determine status and errors
        errors = []
        if reverse_response.get('error'):
            errors.append(reverse_response['error'])
        if isinstance(nearby_response, dict) and nearby_response.get('error'):
            errors.append(nearby_response['error'])
        error_msg = '; '.join(errors) if errors else None

        status = 'success' if location else 'error'
        return {
            'status': status,
            'location': location,
            'pois': pois,
            'error': error_msg
        }

    @classmethod
    @my_deco.retry_geo_api(max_retries=3, delay=1.0)
    def _call_locationiq_reverse(cls, params: Dict) -> Dict:
        endpoint = "https://us1.locationiq.com/v1/reverse"
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            if 'error' in data:
                return {'error': data['error']}
            return data
        except requests.exceptions.RequestException as e:
            return cls._format_error(GeoAPIProvider.LOCATIONIQ, str(e))
        except Exception as e:
            return cls._format_error(GeoAPIProvider.LOCATIONIQ, str(e))

    @classmethod
    @my_deco.retry_geo_api(max_retries=3, delay=1.0)
    def _call_locationiq_nearby(cls, params: Dict) -> Dict:
        endpoint = "https://us1.locationiq.com/v1/nearby"
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return cls._format_error(GeoAPIProvider.LOCATIONIQ, str(e))
        except Exception as e:
            return cls._format_error(GeoAPIProvider.LOCATIONIQ, str(e))

    @classmethod
    def _format_error(cls, provider: GeoAPIProvider, message: str) -> Dict:
        base_error = {
            'status': 'error',
            'error': f"Request failed: {message}",
            'provider': provider.value
        }
        if provider == GeoAPIProvider.AMAP:
            base_error['info'] = base_error.pop('error')
            base_error['infocode'] = '500'
        return base_error

    @staticmethod
    def _standardize_amap_poi(poi: Dict) -> Dict:
        """Standardize AMap POI data to a common format with detailed fields."""
        location = poi.get('location', '').split(',') if 'location' in poi else ['', '']
        return {
            'name': poi.get('name', ''),
            'latitude': location[1] if len(location) > 1 else '',
            'longitude': location[0] if len(location) > 0 else '',
            'distance': poi.get('distance', ''),
            'type': poi.get('type', ''),
            'address': poi.get('address', ''),
            'weight': poi.get('poiweight', ''),
            'class': poi.get('type', '').split(';')[0] if 'type' in poi else '',
            'type': poi.get('type', '').split(';')[0] if 'type' in poi else '',
        }

    @staticmethod
    def _standardize_locationiq_poi(poi: Dict) -> Dict:
        """Standardize LocationIQ POI data to a common format with detailed fields."""
        address = ''
        if 'display_name' in poi:
            address = poi.get('display_name', '')
        elif 'address' in poi:
            address_parts = []
            addr = poi['address']
            if 'name' in addr:
                address_parts.append(addr['name'])
            if 'road' in addr:
                address_parts.append(addr['road'])
            if 'suburb' in addr:
                address_parts.append(addr['suburb'])
            if 'city' in addr:
                address_parts.append(addr['city'])
            if 'state' in addr:
                address_parts.append(addr['state'])
            if 'country' in addr:
                address_parts.append(addr['country'])
            address = ', '.join(filter(None, address_parts))
        return {
            'name': poi.get('name', ''),
            'latitude': poi.get('lat', ''),
            'longitude': poi.get('lon', ''),
            'distance': poi.get('distance', ''),
            'address': address,
            'weight': '',  # LocationIQ does not provide a weight field
            'class': poi.get('class', ''),
            'type': poi.get('type', ''),
        }

    @classmethod
    def set_api_key(cls, provider: GeoAPIProvider, key: str):
        if provider == GeoAPIProvider.AMAP:
            cls.AMAP_API_KEY = key
        elif provider == GeoAPIProvider.LOCATIONIQ:
            cls.LOCATIONIQ_API_KEY = key

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
                                                                datum=MapDatum.WGS84)
            
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

class PhotoInfoExtractor:
    def __init__(self, metadata):
        self.metadata = metadata if isinstance(metadata, dict) else {}

    def get_time_info(self):
        """Extract and format time information from photo metadata"""
        photo_data = self.metadata.get('photo', {})
        create_date = photo_data.get('create_date', '')
        if not create_date:
            return ''

        try:
            # Parse date and time
            date_part, time_part = create_date.split('T')
            year, month, day = date_part.split('-')
            time_part = time_part.split('+')[0]
            hour, minute, _ = time_part.split(':')
            hour_int = int(hour)
            
            # Format date (Chinese format)
            date_str = f"{year}年{int(month)}月{int(day)}日"
            
            # Determine time period
            if 4 <= hour_int < 6:
                period = "清晨"
            elif 6 <= hour_int < 9:
                period = "早晨"
            elif 9 <= hour_int < 11:
                period = "上午"
            elif 11 <= hour_int < 13:
                period = "中午"
            elif 13 <= hour_int < 17:
                period = "下午"
            elif 17 <= hour_int < 19:
                period = "黄昏"
            else:
                period = "夜晚"
            
            # Format time (12-hour format)
            display_hour = hour_int if hour_int <= 12 else hour_int - 12
            if hour_int == 0:
                display_hour = 12
            
            return f"拍摄时间: {date_str} {period}{display_hour}点{minute}分"
        except (IndexError, ValueError, AttributeError):
            return ''

    def get_lens_info(self):
        """Extract and format lens information"""
        lens_data = self.metadata.get('lens', {})
        focal_length = lens_data.get('focal_length_mm')
        if focal_length is None:
            return ''

        if focal_length > 150:
            return f"镜头类型: 超长焦拍摄 ({focal_length}mm)"
        elif focal_length > 70:
            return f"镜头类型: 长焦拍摄 ({focal_length}mm)"
        elif focal_length < 30:
            return f"镜头类型: 广角拍摄 ({focal_length}mm)"
        else:
            return f"镜头类型: 标准人眼视角 ({focal_length}mm)"

    def get_camera_info(self):
        """Extract camera and lens model information"""
        camera_data = self.metadata.get('camera', {})
        camera_info = ''
        if camera_data.get('make') and camera_data.get('model'):
            camera_info = f"相机: {camera_data['make']} {camera_data['model']}"
        
        lens_data = self.metadata.get('lens', {})
        lens_info = ''
        if lens_data.get('model'):
            lens_info = f"镜头型号: {lens_data['model']}"
        
        return '\n'.join(filter(None, [camera_info, lens_info]))

    def get_exposure_info(self):
        """Extract exposure/shutter speed information"""
        photo_data = self.metadata.get('photo', {})
        exposure = photo_data.get('exposure')
        if not exposure:
            return ''

        try:
            numerator, denominator = map(int, exposure.split('/'))
            exposure_value = numerator / denominator
            
            if exposure_value > 1/20:
                return "快门类型: 慢门拍摄"
            elif exposure_value < 1/1000:
                return "快门类型: 高速快门"
        except (ValueError, ZeroDivisionError):
            return ''
        return ''

    def get_aperture_info(self):
        """Extract aperture information"""
        lens_data = self.metadata.get('lens', {})
        aperture = lens_data.get('aperture_value')
        if aperture is None:
            return ''

        if aperture < 2:
            return "光圈类型: 大光圈拍摄"
        elif aperture > 4:
            return "光圈类型: 小光圈拍摄"
        return ''

    def get_geo_info(self):
        """Extract and format geographic information"""
        geo_info = []
        location_parts = []
        gps = self.metadata.get('gps', {})
        if not gps: return ''

        lon = gps.get('longitude_dec')
        lat = gps.get('latitude_dec')
        if not lon or not lat: return ''

        gps_resolved = self.metadata.get('gps_resolved', {})
        if not isinstance(gps_resolved, dict):
            return ''

        gps_converted = gps_resolved.get('conversion_metadata', {})\
                                    .get('converted_coords', {})
        lon = gps_converted.get('lon')
        lat = gps_converted.get('lat')
        geo_info.append(f'GPS={lon:.6f},{lat:.6f}')

        location = gps_resolved.get('location', {})
        components = location.get('components', {})
        
        # Build base location string
        for key in ['country', 'province', 'city', 'district', 'township']:
            if components.get(key):
                location_parts.append(components[key])
        
        if location_parts:
            geo_info.append(f"拍摄地点: {'，'.join(location_parts)}")
        
        if location.get('formatted_address'):
            geo_info.append(f"详细地址: {location['formatted_address']}")
        
        # Process POIs (up to 10)
        pois = gps_resolved.get('pois', [])
        pois = sorted(pois, key=lambda p: float(p.get('distance', '0')))
        if len(pois) > 0:
            geo_info.append("\n附近地点:")
            
            for i, poi in enumerate(pois[:20], 1):
                if not isinstance(poi, dict):
                    continue
                    
                poi_entry = []
                poi_name = poi.get('name', '未命名地点')
                poi_entry.append(f"{i}. {poi_name}")
                
                if 'distance' in poi:
                    poi_entry.append(f"距离: {poi['distance']}米")
                
                poi_type = poi.get('type', '') or poi.get('class', '')
                if poi_type:
                    poi_entry.append(f"类型: {poi_type}")
                
                if poi.get('address'):
                    poi_entry.append(f"地址: {poi['address']}")
                
                geo_info.append(" | ".join(poi_entry))
        
        return '\n'.join(geo_info)

    def get_info(self):
        """Main method to concatenate all extracted information"""
        if not self.metadata:
            return ''
        
        info_parts = [
            self.get_time_info(),
            self.get_lens_info(),
            self.get_camera_info(),
            self.get_exposure_info(),
            self.get_aperture_info(),
            self.get_geo_info()
        ]
        
        return '\n'.join(filter(None, info_parts))


# Example usage
if __name__ == "__main__":
    
    # Can work with either:
    metadata = PhotoMetadataExtractor.extract("./dev_samples/test-sample-cr3/4H4A3790.xmp")
    pprint(metadata)
    
    metadata = PhotoMetadataExtractor.extract("./dev_samples/test-sample-cr3/4H4A3790.cr3")
    pprint(metadata)

    metadata = PhotoMetadataExtractor.extract("./dev_samples/test-sample-cr3/4H4A37901.cr3")
    pprint(metadata)
