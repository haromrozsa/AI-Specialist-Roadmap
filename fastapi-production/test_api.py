"""
Test script for demonstrating valid vs invalid requests.
Run this after starting the API server.
"""
import requests
import json
from sklearn.datasets import load_digits

BASE_URL = "http://127.0.0.1:8000"


def print_result(name: str, response: requests.Response):
    """Pretty print API response."""
    print(f"\n{'=' * 60}")
    print(f"🧪 TEST: {name}")
    print(f"{'=' * 60}")
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")


def main():
    print("\n" + "🚀 API TEST SUITE ".center(60, "="))

    # ─────────────────────────────────────────────────────────
    # Test 1: Health Check
    # ─────────────────────────────────────────────────────────
    r = requests.get(f"{BASE_URL}/")
    print_result("Health Check", r)

    # ─────────────────────────────────────────────────────────
    # Test 2: Valid Request - Single Sample
    # ─────────────────────────────────────────────────────────
    digits = load_digits()
    sample = digits.data[0].tolist()  # First digit (should be 0)

    r = requests.post(
        f"{BASE_URL}/predict",
        json={"data": [sample]}
    )
    print_result(f"Valid Request - Single Sample (actual digit: {digits.target[0]})", r)

    # ─────────────────────────────────────────────────────────
    # Test 3: Valid Request - Batch
    # ─────────────────────────────────────────────────────────
    samples = digits.data[:5].tolist()
    actual_labels = digits.target[:5].tolist()

    r = requests.post(
        f"{BASE_URL}/predict",
        json={"data": samples}
    )
    print_result(f"Valid Request - Batch of 5 (actual: {actual_labels})", r)

    # ─────────────────────────────────────────────────────────
    # Test 4: Invalid - Wrong Feature Count
    # ─────────────────────────────────────────────────────────
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"data": [[1.0, 2.0, 3.0]]}  # Only 3 features instead of 64
    )
    print_result("Invalid Request - Wrong Feature Count (3 instead of 64)", r)

    # ─────────────────────────────────────────────────────────
    # Test 5: Invalid - Feature Value Out of Range
    # ─────────────────────────────────────────────────────────
    bad_sample = [100.0] * 64  # Values should be 0-16
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"data": [bad_sample]}
    )
    print_result("Invalid Request - Feature Values Out of Range (100 > 16)", r)

    # ─────────────────────────────────────────────────────────
    # Test 6: Invalid - Empty Data
    # ─────────────────────────────────────────────────────────
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"data": []}
    )
    print_result("Invalid Request - Empty Data Array", r)

    # ─────────────────────────────────────────────────────────
    # Test 7: Invalid - Missing 'data' Field
    # ─────────────────────────────────────────────────────────
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"samples": [[0.0] * 64]}  # Wrong key
    )
    print_result("Invalid Request - Missing 'data' Field", r)

    # ─────────────────────────────────────────────────────────
    # Test 8: Invalid - Malformed JSON
    # ─────────────────────────────────────────────────────────
    r = requests.post(
        f"{BASE_URL}/predict",
        data="not valid json",
        headers={"Content-Type": "application/json"}
    )
    print_result("Invalid Request - Malformed JSON", r)

    print("\n" + " TEST SUITE COMPLETE ".center(60, "=") + "\n")


if __name__ == "__main__":
    main()