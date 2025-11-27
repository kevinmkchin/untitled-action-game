#ifndef SHOOTER_KEYBOARD_H
#define SHOOTER_KEYBOARD_H

enum shooter_kb_key : u8
{
    KB_BACKSPACE = 0x08,
    KB_TAB = 0x09,
    KB_RETURN = 0x0D,
    KB_ESC = 0x1B,
    KB_END = 0x23,
    KB_HOME = 0x24,
    KB_INS = 0x2D,
    KB_DEL = 0x2E,
    KB_LEFT = 0x25,
    KB_UP = 0x26,
    KB_RIGHT = 0x27,
    KB_DOWN = 0x28,
    KB_0 = 0x30,
    KB_1 = 0x31,
    KB_2 = 0x32,
    KB_3 = 0x33,
    KB_4 = 0x34,
    KB_5 = 0x35,
    KB_6 = 0x36,
    KB_7 = 0x37,
    KB_8 = 0x38,
    KB_9 = 0x39,
    KB_A = 0x41,
    KB_B = 0x42,
    KB_C = 0x43,
    KB_D = 0x44,
    KB_E = 0x45,
    KB_F = 0x46,
    KB_G = 0x47,
    KB_H = 0x48,
    KB_I = 0x49,
    KB_J = 0x4A,
    KB_K = 0x4B,
    KB_L = 0x4C,
    KB_M = 0x4D,
    KB_N = 0x4E,
    KB_O = 0x4F,
    KB_P = 0x50,
    KB_Q = 0x51,
    KB_R = 0x52,
    KB_S = 0x53,
    KB_T = 0x54,
    KB_U = 0x55,
    KB_V = 0x56,
    KB_W = 0x57,
    KB_X = 0x58,
    KB_Y = 0x59,
    KB_Z = 0x5A,
    KB_F1 = 0x70,
    KB_F2 = 0x71,
    KB_F3 = 0x72,
    KB_F4 = 0x73,
    KB_F5 = 0x74,
    KB_F6 = 0x75,
    KB_F7 = 0x76,
    KB_F8 = 0x77,
    KB_F9 = 0x78,
    KB_F10 = 0x79,
    KB_F11 = 0x7A,
    KB_F12 = 0x7B,
    KB_PLUS = 0xBB, // Equals and Plus key
    KB_MINUS = 0xBD, // Dash and Underscore key
    KB_COMMA = 0xBC, // Comma and Less Than key
    KB_PERIOD = 0xBE, // Period and Greater Than key
    KB_ANSI_LBRACE = 0xDB, // It can vary by keyboard. For the US ANSI keyboard, the Left Brace key
    KB_ANSI_RBRACE = 0xDD, // It can vary by keyboard. For the US ANSI keyboard, the Right Brace key
    KB_ANSI_FORWARDSLASH = 0xBF, // It can vary by keyboard. For the US ANSI keyboard, the Forward Slash and Question Mark key
    KB_ANSI_BACKSLASH = 0xDC, // It can vary by keyboard. For the US ANSI keyboard, the Backslash and Pipe key
    KB_ANSI_APOSTROPHE = 0xDE, // It can vary by keyboard. For the US ANSI keyboard, the Apostrophe and Double Quotation Mark key
    KB_ANSI_TILDE = 0xC0, // It can vary by keyboard. For the US ANSI keyboard, the Grave Accent and Tilde key

    KB_KEYS_MAX = 0xE0
};

#endif // SHOOTER_KEYBOARD_H
