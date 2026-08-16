from pynq import Overlay

BITSTREAM_PATH = "/home/xilinx/pynq/overlays/dqn_eva/dqn_eva.bit"

def main():
    print("Loading overlay...")
    overlay = Overlay(BITSTREAM_PATH)

    print("\nOverlay loaded successfully.")

    print("\nAvailable IP blocks:")
    for name in overlay.ip_dict:
        print("  -", name)

    if hasattr(overlay, "dqn_eva_0"):
        dqn_ip = overlay.dqn_eva_0
        print("\nRegister map for dqn_eva_0:")
        print(dqn_ip.register_map)
    else:
        print("\nERROR: dqn_ev_0 was not found.")
        print("Check the IP name printed above.")

if __name__ == "__main__":
    main()