import pandas as pd
import yfinance as yf
import numpy as np
import os
import time
import argparse
import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# 導入核心掃描模組
import scan_module 

# --- 全域常數 ---
HOLD_LIST_FILE = "hold_list.txt"
MAX_WORKERS = 10 # 多線程處理的線程數

# --- 主程式 ---
if __name__ == '__main__':
    # 配置標準輸出以使用 UTF-8 編碼
    if sys.stdout.encoding != 'UTF-8':
        sys.stdout.reconfigure(encoding='UTF-8')

    parser = argparse.ArgumentParser(description="自動股票掃描與報告工具")
    parser.add_argument(
        "--model", 
        type=str, 
        default="Model_conf",
        help="要使用的單一模型設定檔 (預設: Model_conf)。可選: Model_conf_alt, Model_conf_aggressive"
    )
    parser.add_argument(
        "--strategy_type", 
        type=str, 
        choices=['conservative', 'aggressive'], 
        default='conservative',
        help="單一模型分析時的策略類型 (預設: conservative)。"
    )
    parser.add_argument(
        "--extend", 
        action="store_true", # 設置為 True 則表示啟用雙模型分析
        help="啟用後將同時使用 Model_conf 和 Model_conf_alt 進行雙模型判斷。"
    )
    args = parser.parse_args()
    
    start_time = time.time()
    
    # 獲取動態生成的掃描清單 (現在從 scan_module 獲取) 
    scan_list_dynamic = scan_module.get_dynamic_scan_list() 
    held_tickers = scan_module.read_hold_list()
    
    # 最終的掃描清單是動態清單和持股清單的聯集
    all_tickers_to_scan = sorted(list(set(scan_list_dynamic) | held_tickers))
    
    print(f"===== 自動股票掃描系統啟動 ({time.strftime('%Y-%m-%d %H:%M:%S')}) =====")
    if args.extend:
        print(f"分析模式: 💎 雙模型綜合判斷 (Model_conf & Model_conf_alt)")
        # 雙模型模式下，策略類型由各自模型決定，此處不需額外參數
        model1_config = scan_module.load_config("Model_conf")
        model2_config = scan_module.load_config("Model_conf_alt")
        model1_strategy_type = 'conservative' # Model_conf 預設為保守
        model2_strategy_type = 'conservative' # Model_conf_alt 預設為保守
    else:
        print(f"分析模式: 單一模型 ({args.model}, 策略: {args.strategy_type})")
        single_model_config = scan_module.load_config(args.model)
    
    print(f"掃描標的總數: {len(all_tickers_to_scan)}")

    all_results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for ticker in all_tickers_to_scan:
            if args.extend:
                # 提交兩個分析任務
                futures[executor.submit(scan_module.analyze_ticker, ticker, model1_config, model1_strategy_type)] = (ticker, 'model1')
                futures[executor.submit(scan_module.analyze_ticker, ticker, model2_config, model2_strategy_type)] = (ticker, 'model2')
            else:
                # 提交單一分析任務
                futures[executor.submit(scan_module.analyze_ticker, ticker, single_model_config, args.strategy_type)] = (ticker, 'single')
        
        # 處理結果
        processed_tickers = set()
        for i, future in enumerate(as_completed(futures)):
            ticker_info, model_id = futures[future]
            ticker = ticker_info[0] if isinstance(ticker_info, tuple) else ticker_info # 處理多模型傳遞方式
            
            try:
                result_tuple = future.result() # result_tuple: (ticker, result_dict, status_str)
                if result_tuple[1]: # 如果分析成功
                    if args.extend:
                        if ticker not in all_results:
                            all_results[ticker] = {'model1': None, 'model2': None}
                        all_results[ticker][model_id] = result_tuple[1] # 存儲單一模型的結果
                    else:
                        all_results[ticker] = result_tuple[1]
                else:
                    all_results[ticker] = None # 表示該股票分析失敗

            except Exception as e:
                print(f"\r處理 {ticker} 時發生錯誤: {e}", end="") # 錯誤時打印，不影響進度條
                all_results[ticker] = None

            # 打印進度條 (針對每個股票只更新一次)
            if ticker not in processed_tickers: # 確保每個 ticker 只計數一次
                processed_tickers.add(ticker)
                progress = len(processed_tickers) / len(all_tickers_to_scan)
                print(f"\r進度: ([{'=' * int(progress * 20):<20}] {progress:.1%}) - 處理 {ticker}", end="")

    print("\n\n" + "="*50)
    print("      💼 您持有的股票狀態報告")
    print("="*50)
    
    if not held_tickers:
        print("持股清單是空的。")
    else:
        for ticker in sorted(list(held_tickers)):
            res_data = all_results.get(ticker)
            if res_data and res_data != None: # 確保分析成功
                if args.extend:
                    model1_res = res_data.get('model1')
                    model2_res = res_data.get('model2')
                    if model1_res and model2_res:
                        score1 = scan_module.get_recommendation_score(model1_res['barometer'], model1_res['recovery'])
                        score2 = scan_module.get_recommendation_score(model2_res['barometer'], model2_res['recovery'])
                        final_score = scan_module.get_combined_recommendation_score(score1, score2)
                        final_verbal = scan_module.get_final_verbal_score(final_score)
                        print(f"  - {ticker:<10} | 價格: {model1_res['price']:>8} | 建議: {final_verbal} (M1: {model1_res['recommendation']}, M2: {model2_res['recommendation']})")
                    else:
                        print(f"  - {ticker:<10} | 無法獲取雙模型分析結果。")
                else:
                    if res_data['barometer'] != "資料不足" and "指標計算後數據不足" not in res_data['recommendation']:
                        print(f"  - {res_data['ticker']:<10} | 價格: {res_data['price']:>8} | 狀態: {res_data['barometer']:<15} | 建議: {res_data['recommendation']}")
                    else:
                        print(f"  - {ticker:<10} | 無法獲取有效分析結果。")
            else:
                print(f"  - {ticker:<10} | 無法獲取分析結果。")

    print("\n" + "="*50)
    print("      🔍 市場潛在機會掃描")
    print("="*50)
    
    opportunities = []
    if args.extend:
        for ticker, res_data in all_results.items():
            if ticker in held_tickers or res_data is None: continue # 跳過持股和分析失敗的
            model1_res = res_data.get('model1')
            model2_res = res_data.get('model2')
            if model1_res and model2_res and "資料不足" not in model1_res['barometer'] and "指標計算後數據不足" not in model1_res['recommendation']:
                score1 = scan_module.get_recommendation_score(model1_res['barometer'], model1_res['recovery'])
                score2 = scan_module.get_recommendation_score(model2_res['barometer'], model2_res['recovery'])
                final_score = scan_module.get_combined_recommendation_score(score1, score2)
                final_verbal = scan_module.get_final_verbal_score(final_score)
                if final_score != 0: # 只顯示有買賣建議的
                    opportunities.append({
                        "ticker": ticker,
                        "price": model1_res['price'],
                        "recommendation": final_verbal,
                        "model1_rec": model1_res['recommendation'],
                        "model2_rec": model2_res['recommendation']
                    })
    else: # 單一模型模式
        opportunities = [res for res in all_results.values() 
                         if res and res['ticker'] not in held_tickers 
                         and ("建議進場" in res['recommendation'] or "建議出場" in res['recommendation'])
                         and res['barometer'] != "資料不足" and "指標計算後數據不足" not in res['recommendation']]
    
    if not opportunities:
        print("在掃描清單中未發現新的進出場機會。")
    else:
        # 分類顯示
        buy_ops = [op for op in opportunities if "買入" in op['recommendation']]
        sell_ops = [op for op in opportunities if "賣出" in op['recommendation'] or "減碼" in op['recommendation']]
        
        if buy_ops:
            print("\n  --- 🟢 潛在進場機會 ---")
            for op in sorted(buy_ops, key=lambda x: x['recommendation'], reverse=True): # 強力買入優先
                 if args.extend:
                     print(f"  - {op['ticker']:<10} | 價格: {op['price']:>8} | 綜合建議: {op['recommendation']} (M1: {op['model1_rec']}, M2: {op['model2_rec']})")
                 else:
                     print(f"  - {op['ticker']:<10} | 價格: {op['price']:>8} | 建議: {op['recommendation']}")
        
        if sell_ops:
            print("\n  --- 🔴 潛在出場/減碼機會 ---")
            for op in sorted(sell_ops, key=lambda x: x['recommendation']): # 強力賣出優先
                 if args.extend:
                     print(f"  - {op['ticker']:<10} | 價格: {op['price']:>8} | 綜合建議: {op['recommendation']} (M1: {op['model1_rec']}, M2: {op['model2_rec']})")
                 else:
                     print(f"  - {op['ticker']:<10} | 價格: {op['price']:>8} | 建議: {op['recommendation']}")

    end_time = time.time()
    print(f"\n\n掃描完成，總耗時: {end_time - start_time:.2f} 秒。")