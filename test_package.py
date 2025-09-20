#!/usr/bin/env python3
"""
Basic tests for the WhisperX pipeline package structure.
These tests verify the package can be imported and basic functionality works
without requiring the full dependency stack.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_package_import():
    """Test that the package can be imported."""
    try:
        import whisper_pipeline
        print("✅ Package import successful")
        return True
    except Exception as e:
        print(f"❌ Package import failed: {e}")
        return False

def test_config_creation():
    """Test configuration creation without dependencies."""
    try:
        from whisper_pipeline.config import PipelineConfig, get_default_config, SpeakerPosition
        
        # Test default config
        config = get_default_config()
        assert config is not None
        assert hasattr(config, 'transcription')
        assert hasattr(config, 'diarization')
        assert hasattr(config, 'spatialization')
        assert hasattr(config, 'output')
        
        # Test speaker position
        pos = SpeakerPosition(x=1.0, y=2.0, z=0.0)
        assert pos.x == 1.0
        assert pos.y == 2.0
        assert pos.z == 0.0
        
        print("✅ Configuration creation successful")
        return True
    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")
        return False

def test_config_yaml_serialization():
    """Test YAML serialization/deserialization."""
    try:
        from whisper_pipeline.config import PipelineConfig, get_default_config
        
        config = get_default_config()
        
        # Test YAML export/import
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            
            # Test loading
            loaded_config = PipelineConfig.from_yaml(f.name)
            assert loaded_config.transcription.model_name == config.transcription.model_name
            
            # Clean up
            os.unlink(f.name)
        
        print("✅ YAML serialization successful")
        return True
    except Exception as e:
        print(f"❌ YAML serialization failed: {e}")
        return False

def test_cli_import():
    """Test CLI module import."""
    try:
        from whisper_pipeline import cli
        print("✅ CLI import successful")
        return True
    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        return False

def test_utils_functions():
    """Test utility functions that don't require audio dependencies."""
    try:
        from whisper_pipeline.utils import (
            setup_logging, _format_time, _format_srt_time,
            calculate_3d_distance, get_file_extension, ensure_output_directory
        )
        
        # Test logging setup
        logger = setup_logging(verbose=False)
        assert logger is not None
        
        # Test time formatting
        assert _format_time(3661) == "01:01:01"
        assert _format_time(61) == "01:01"
        assert _format_srt_time(3661.5) == "01:01:01,500"
        
        # Test 3D distance calculation
        dist = calculate_3d_distance((0, 0, 0), (3, 4, 0))
        assert abs(dist - 5.0) < 0.001
        
        # Test file extensions
        assert get_file_extension('wav') == '.wav'
        assert get_file_extension('json') == '.json'
        
        # Test directory creation
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, 'subdir', 'file.txt')
            ensure_output_directory(test_path)
            assert os.path.exists(os.path.dirname(test_path))
        
        print("✅ Utility functions successful")
        return True
    except Exception as e:
        print(f"❌ Utility functions failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_package_structure():
    """Test package structure and files."""
    try:
        package_dir = Path(__file__).parent / "whisper_pipeline"
        
        print(f"Checking package directory: {package_dir}")
        
        # Check required files exist
        required_files = [
            "__init__.py",
            "config.py", 
            "utils.py",
            "pipeline.py",
            "transcription.py",
            "diarization.py",
            "spatialization.py",
            "cli.py"
        ]
        
        for file in required_files:
            file_path = package_dir / file
            if not file_path.exists():
                print(f"Missing required file: {file_path}")
                return False
        
        print("✅ Package structure valid")
        return True
    except Exception as e:
        print(f"❌ Package structure validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("WhisperX Pipeline - Package Tests")
    print("=" * 40)
    
    tests = [
        test_package_import,
        test_package_structure,
        test_config_creation,
        test_config_yaml_serialization,
        test_utils_functions,
        test_cli_import,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())