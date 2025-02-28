import sys

def binary_to_cpp_array(input_file, output_file, array_name="OptixIRBinary"):
    with open(input_file, "rb") as f:
        data = f.read()

    with open(output_file, "w") as f:
        f.write(f"#pragma once\n")
        f.write(f"constexpr unsigned char {array_name}[] = {{\n")
        for i, byte in enumerate(data):
            f.write(f" 0x{byte:02X},")
            if (i + 1) % 12 == 0:  # Wrap lines every 12 bytes
                f.write("\n")
        f.write("\n};\n\n")
        f.write(f"constexpr size_t {array_name}Size = sizeof({array_name});\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 bin_to_cpp.py <input_binary> <output_cpp>")
    else:
        binary_to_cpp_array(sys.argv[1], sys.argv[2])
