import argparse
import csv
import os
import re
import serial
import sys
from datetime import datetime

# =========================================================
# SYSTEM CONFIGURATION ALIGNMENT
# =========================================================
# Khóa cứng tốc độ 2,000,000 bps đồng bộ tuyệt đối với ESP32 IIR_Collection.ino
DEFAULT_BAUD = 921600
DEFAULT_PORT = "COM5"  # Thay thành '/dev/ttyUSB0' nếu chạy trên Linux/macOS


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="High-speed Serial Logger for Arduino/ESP32 Data Collection."
    )
    parser.add_argument(
        "-p",
        "--port",
        default=DEFAULT_PORT,
        help=f"Serial port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "-b",
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Baud rate (default: {DEFAULT_BAUD})",
    )
    parser.add_argument(
        "-l",
        "--label",
        default="imbalanced",
        help="Label for the data collection session (default: obstruction)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Định nghĩa cấu trúc file CSV đích
    # Khớp hoàn toàn với REQUIRE_COLUMNS trong config.py và segmenter.py
    filename = f"arduino_serial_data_{args.label}.csv"
    headers = ["timestamp_ms", "accX_raw", "accY_raw", "accZ_raw", "accZ_filt", "label"]

    file_exists = os.path.isfile(filename)

    print("=" * 70)
    print("HIGH-SPEED SERIAL LOGGER STARTED")
    print("=" * 70)
    print(f"Target Port          : {args.port}")
    print(f"Baud Rate            : {args.baud} bps")
    print(f"Session Label        : {args.label}")
    print(f"Output File          : {filename}")
    print("Press Ctrl+C to stop logging safely.")
    print("=" * 70 + "\n")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1.0)
        ser.flushInput()  # Giải phóng dữ liệu cũ còn tồn dư trong buffer phần cứng
    except serial.SerialException as e:
        print(f"[CRITICAL ERROR] Cannot open port {args.port}: {e}")
        sys.exit(1)

    # 🛠️ SỬA LOGIC REGEX: Mở rộng khớp chính xác 5 trường số thực (4 dấu phẩy)
    # Định dạng: timestamp, accX_raw, accY_raw, accZ_raw, accZ_filt
    data_pattern = re.compile(
        r"^\d+(\.\d+)?,(-?\d+(\.\d+)?),(-?\d+(\.\d+)?),(-?\d+(\.\d+)?),(-?\d+(\.\d+)?)$"
    )

    try:
        with open(filename, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)

            # Chèn dòng Header tiêu đề chuẩn hóa nếu file mới được tạo lập
            if not file_exists:
                writer.writerow(headers)
                csv_file.flush()

            sample_count = 0

            while True:
                if ser.in_waiting > 0:
                    try:
                        line = ser.readline().decode("utf-8", errors="ignore").strip()
                    except Exception:
                        continue

                    if not line:
                        continue

                    # Nếu ESP32 gửi chuỗi chứa text của tiêu đề cũ hoặc rác lúc khởi động thì bỏ qua
                    if "timestamp_ms" in line or "accX" in line:
                        continue

                    # Tách chuỗi văn bản nhận được
                    # Chuỗi thô từ ESP32 chứa cả nhãn ở cuối: "timestamp,x,y,z,zf,label"
                    parts = line.split(",")
                    
                    # Trích xuất 5 trường số đầu tiên để kiểm tra định dạng qua Regex
                    if len(parts) >= 5:
                        sensor_data_str = ",".join(parts[:5])
                        
                        if data_pattern.match(sensor_data_str):
                            # Lấy chuẩn dữ liệu số
                            row_data = parts[:5]
                            # Gắn thẻ nhãn đồng bộ từ tham số CLI truyền vào thay vì tin tưởng nhãn từ ESP32
                            row_data.append(args.label)

                            writer.writerow(row_data)
                            sample_count += 1

                            # Xử lý flush buffer định kỳ 500 mẫu nhằm tránh nghẽn I/O ổ cứng
                            if sample_count % 500 == 0:
                                csv_file.flush()
                                print(
                                    f"[{datetime.now().strftime('%H:%M:%S')}] Captured {sample_count} samples...",
                                    end="\r",
                                )
                        else:
                            # Nếu chuỗi không khớp cấu trúc số, in thẳng ra terminal để debug lỗi hệ thống/I2C từ ESP32
                            print(f"\n[ESP32 System Msg]: {line}")
                    else:
                        print(f"\n[Malformed Packet]: {line}")

    except KeyboardInterrupt:
        print(f"\n\nLogging stopped by user. Total valid samples saved: {sample_count}")
    finally:
        if "ser" in locals() and ser.is_open:
            ser.close()
            print("Serial port closed safely.")


if __name__ == "__main__":
    main()