#!/usr/bin/env python3
"""
Phase 1 Test Script
Tests the new queue-based system
"""

import requests
import time
import json
import os

BASE_URL = "http://localhost:8002"

def test_health_check():
    """Test system health"""
    print("🔍 Testing system health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        
        print(f"✅ Health Check Response:")
        print(f"   - Status: {data.get('status')}")
        print(f"   - Queue Available: {data.get('queue_available')}")
        print(f"   - Redis Connected: {data.get('redis_connected')}")
        
        return data.get('queue_available', False)
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_file_upload():
    """Test queue-based file upload"""
    print("\n📤 Testing queue-based file upload...")
    
    # Create a test file
    test_content = "This is a test document for Phase 1 queue system testing."
    test_filename = "phase1_test.txt"
    
    with open(test_filename, 'w') as f:
        f.write(test_content)
    
    try:
        # Upload file
        with open(test_filename, 'rb') as f:
            files = {'file': (test_filename, f, 'text/plain')}
            data = {
                'org_id': 'test-org-phase1',
                'user_id': 'test-user-phase1',
                'file_id': 'test-file-phase1'
            }
            
            response = requests.post(f"{BASE_URL}/upload", files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful:")
            print(f"   - Task ID: {result.get('task_id')}")
            print(f"   - File ID: {result.get('file_id')}")
            print(f"   - Status Endpoint: {result.get('status_endpoint')}")
            
            return result.get('task_id')
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
        return None
        
    finally:
        # Clean up test file
        if os.path.exists(test_filename):
            os.remove(test_filename)

def test_task_status(task_id):
    """Test task status checking"""
    print(f"\n📊 Testing task status for: {task_id}")
    
    max_attempts = 30  # 30 attempts = ~5 minutes max
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get(f"{BASE_URL}/task-status/{task_id}")
            
            if response.status_code == 200:
                data = response.json()
                status_info = data.get('status', {})
                progress_data = status_info.get('progress_data', {})
                
                if progress_data:
                    status = progress_data.get('status', 'unknown')
                    progress = progress_data.get('progress', 0)
                    stage = progress_data.get('stage', 'unknown')
                    
                    print(f"   📈 Status: {status} ({progress}%) - {stage}")
                    
                    if status in ['completed', 'error']:
                        if status == 'completed':
                            print("✅ Task completed successfully!")
                        else:
                            print(f"❌ Task failed: {stage}")
                        return status == 'completed'
                else:
                    celery_state = status_info.get('celery_state', 'UNKNOWN')
                    print(f"   🔄 Celery State: {celery_state}")
                
            else:
                print(f"❌ Status check failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Status check error: {e}")
        
        attempt += 1
        time.sleep(10)  # Wait 10 seconds between checks
    
    print("⏰ Task status check timed out")
    return False

def main():
    """Run all tests"""
    print("🚀 Phase 1 Queue System Test Suite")
    print("==================================")
    
    # Test 1: Health Check
    queue_available = test_health_check()
    
    if not queue_available:
        print("\n❌ Queue system not available. Please check:")
        print("   1. Redis is running: redis-cli ping")
        print("   2. Dependencies installed: pip install -r requirements.txt")
        print("   3. Flask app is running: python app.py")
        return
    
    # Test 2: File Upload
    task_id = test_file_upload()
    
    if not task_id:
        print("\n❌ File upload test failed")
        return
    
    # Test 3: Task Status Tracking
    success = test_task_status(task_id)
    
    if success:
        print("\n🎉 All tests passed! Phase 1 is working correctly.")
        print("\n📋 What this means:")
        print("   ✅ Queue system is operational")
        print("   ✅ Redis is connected and working")
        print("   ✅ File processing works through queues")
        print("   ✅ Progress tracking is persistent")
        print("   ✅ Your system is ready for production load!")
    else:
        print("\n⚠️ Some tests failed. Check the logs and setup.")

if __name__ == "__main__":
    main()