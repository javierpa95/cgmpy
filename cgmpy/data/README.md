# Refactored Data Module

## Overview

The `cgmpy.data` module has been completely refactored to provide a modular and maintainable architecture for glucose data handling. The refactoring splits responsibilities into specialized modules while maintaining backward compatibility.

## Modular Architecture

### 🔧 Specialized Modules

#### 1. `DataLoader` (`loader.py`)
- **Responsibility**: Data loading from different sources
- **Features**:
  - Loading from CSV, Parquet, and DataFrames
  - Automatic delimiter detection
  - Robust error handling
  - Support for different header formats

#### 2. `DataProcessor` (`processor.py`)
- **Responsibility**: Data processing and validation
- **Features**:
  - Optimized path for Parquet files
  - Column validation
  - Data type conversion
  - Duplicate handling
  - Null data cleaning
  - Date filtering

#### 3. `DataAnalyzer` (`analyzer.py`)
- **Responsibility**: Basic data analysis
- **Features**:
  - Typical interval calculation
  - Basic data information
  - Disconnection analysis
  - Data quality metrics
  - Summary generation

#### 4. `DataExporter` (`exporter.py`)
- **Responsibility**: Data export
- **Features**:
  - Export to optimized Parquet
  - Export to CSV and Excel
  - Appending to existing files
  - Duplicate handling
  - Data type optimization

### 📊 Main Class: `ModularGlucoseData`

The `ModularGlucoseData` class (`core.py`) integrates all specialized modules providing a clean and functional interface:

```python
from cgmpy.data import ModularGlucoseData

# Create instance
data = ModularGlucoseData('data.csv')

# Get information
info = data.info()
print(f"Data records: {info['n_records']}")
print(f"Typical interval: {info['typical_interval']:.1f} minutes")

# Filter data
filtered = data.filter_by_date_range('2024-01-01', '2024-01-31')
glucose_filtered = data.filter_by_glucose_range(70, 180)

# Export
data.to_parquet('optimized_data.parquet')
data.to_csv('exported_data.csv')
```

## Specialized Classes by Device

### 🔬 Supported Devices

#### 1. `Dexcom` - Dexcom Clarity
```python
from cgmpy.data import Dexcom
dexcom = Dexcom('dexcom_data.csv')
```

#### 2. `Libreview` - FreeStyle Libre
```python
from cgmpy.data import Libreview
libreview = Libreview('libreview_data.csv', header=2)
```

#### 3. `MedtronicCarelink` - Medtronic CareLink
```python
from cgmpy.data import MedtronicCarelink
medtronic = MedtronicCarelink('medtronic_data.csv')
```

#### 4. `TandemDiabetes` - Tandem Diabetes
```python
from cgmpy.data import TandemDiabetes
tandem = TandemDiabetes('tandem_data.csv')
```

### 🤖 Automatic Detection

```python
from cgmpy.data import detect_device_type, create_specialized_loader

# Detect device type
device_type = detect_device_type('data.csv')
print(f"Detected device: {device_type}")

# Create appropriate loader automatically
loader = create_specialized_loader('data.csv')
```

## Backward Compatibility

The refactoring maintains **100% backward compatibility**:

```python
# Existing code still works
from cgmpy import GlucoseData, Dexcom, Libreview

# GlucoseData is now an alias for ModularGlucoseData
data = GlucoseData('data.csv')
info = data.info()
```

## Benefits of Refactoring

### 🎯 Single Responsibility Principle
- Each module has a specific responsibility
- Easy maintenance and testing
- Cleaner and more understandable code

### 📈 Improved Performance
- Optimized path for Parquet files
- Efficient processing of large data
- Optimized calculations with NumPy

### 🔧 Extensibility
- Easy to add new device types
- Independent and reusable modules
- Architecture prepared for future improvements

### 🧪 Testability
- Each module can be tested independently
- Clear separation of responsibilities
- Easier mocking for unit tests

## Usage Examples

### Basic Usage
```python
from cgmpy.data import ModularGlucoseData

# Load data
data = ModularGlucoseData('data.csv', log=True)

# Get information
print(data)
info = data.info(include_disconnections=True)

# Access data
raw_data = data.get_raw_data()
glucose_values = data.get_glucose_values()
timestamps = data.get_timestamps()
```

### Advanced Usage
```python
from cgmpy.data import DataLoader, DataProcessor, DataAnalyzer

# Direct modular usage
loader = DataLoader()
processor = DataProcessor()
analyzer = DataAnalyzer()

# Process step by step
raw_data = loader.load_from_source('data.csv', 'time', 'glucose')
processed_data, time_diffs = processor.process_data(raw_data, 'time', 'glucose')
typical_interval = analyzer.calculate_typical_interval(time_diffs)
```

### Filtering and Analysis
```python
# Filter by date
january_data = data.filter_by_date_range('2024-01-01', '2024-01-31')

# Filter by glucose
normal_range = data.filter_by_glucose_range(70, 180)

# Quality metrics
quality = data.get_data_quality_metrics()
print(f"Total gaps detected: {quality['total_gaps']}")
print(f"Maximum gap: {quality['max_gap_hours']:.1f} hours")
```

## Migration from Previous Version

### ✅ No Changes Required
Existing code continues to work without modifications.

### 🆕 To Take Advantage of New Features
```python
# Before
from cgmpy import GlucoseData
data = GlucoseData('data.csv')

# Now (optional)
from cgmpy.data import ModularGlucoseData
data = ModularGlucoseData('data.csv')

# New features
filtered = data.filter_by_glucose_range(80, 180)
quality = data.get_data_quality_metrics()
```

## Logging and Performance

```python
# Enable detailed logging
data = ModularGlucoseData('data.csv', log=True)

# Logging shows:
# - Load times
# - Processing operations
# - Optimizations applied
# - Memory usage
```

## Next Steps

The data module refactoring lays the foundation for:

1. **Metrics refactoring** - Split `glucose_metrics.py`
2. **Plotting refactoring** - Split `glucose_plot.py`
3. **Analysis modules** - Move `glucose_pregnancy.py`
4. **Unit tests** - Create a complete test suite
5. **Documentation** - Complete technical documentation

---

*This refactoring reduces the original file from 791 lines to 6 specialized modules, significantly improving the maintainability and extensibility of the code.*
