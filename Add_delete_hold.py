import sys
import os

HOLD_LIST_FILE = "hold_list.txt"

def read_hold_list():
    """讀取持股清單，返回一個包含股票代碼的集合(set)。"""
    if not os.path.exists(HOLD_LIST_FILE):
        return set()
    with open(HOLD_LIST_FILE, 'r', encoding='utf-8') as f:
        # 移除空白行並轉換為大寫
        tickers = {line.strip().upper() for line in f if line.strip()}
    return tickers

def write_hold_list(tickers):
    """將股票代碼集合寫回文件，每行一個。"""
    sorted_tickers = sorted(list(tickers))
    with open(HOLD_LIST_FILE, 'w', encoding='utf-8') as f:
        for ticker in sorted_tickers:
            f.write(ticker + '\n')

def print_usage():
    """打印使用說明。"""
    print("\n持股清單管理工具")
    print("用法:")
    print("  python Add_delete_hold.py add <股票代碼>  - 新增一檔股票")
    print("  python Add_delete_hold.py del <股票代碼>  - 刪除一檔股票")
    print("  python Add_delete_hold.py list              - 顯示目前清單")
    print("\n範例:")
    print("  python Add_delete_hold.py add AAPL")
    print("  python Add_delete_hold.py del 2330.TW")

def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()
    
    current_tickers = read_hold_list()

    if command == 'list':
        if not current_tickers:
            print("持股清單是空的。")
        else:
            print("當前持股清單:")
            for ticker in sorted(list(current_tickers)):
                print(f"- {ticker}")
        return

    if len(sys.argv) != 3:
        print("錯誤: 'add' 和 'del' 指令需要一個股票代碼 ويعمل.")
        print_usage()
        sys.exit(1)

    ticker_to_modify = sys.argv[2].upper()

    if command == 'add':
        if ticker_to_modify in current_tickers:
            print(f"'{ticker_to_modify}' 已經在清單中。")
        else:
            current_tickers.add(ticker_to_modify)
            write_hold_list(current_tickers)
            print(f"成功新增 '{ticker_to_modify}' 到持股清單。")
    
    elif command == 'del' or command == 'delete':
        if ticker_to_modify in current_tickers:
            current_tickers.remove(ticker_to_modify)
            write_hold_list(current_tickers)
            print(f"成功從持股清單中刪除 '{ticker_to_modify}'。")
        else:
            print(f"'{ticker_to_modify}' 不在持股清單中，無需刪除。")
            
    else:
        print(f"錯誤: 無效的指令 '{command}' ويعمل.")
        print_usage()

if __name__ == '__main__':
    main()
