#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Linq;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    /// <summary>
    /// Lanto NQ Quantitative Strategy — CSV Signal Reader variant.
    /// VERSION: v1.0-CSV (2026-04-02)
    /// Reads pre-computed signals from an external CSV file (Python output).
    /// All signal detection is done in Python; NT8 handles execution and trade management only.
    /// Trade management logic is identical to LantoNQStrategy v8.1.
    /// </summary>
    public class LantoNQStrategy_CSV : Strategy
    {
        #region Data Classes

        private class TradeState
        {
            public int Direction;
            public double EntryPrice;
            public double StopPrice;
            public double TP1Price;
            public int Contracts;
            public int OrigContracts;
            public DateTime EntryTime;
            public int EntryBarIdx;
            public bool Trimmed;
            public string SignalType;
            public double OrigStopDist;
            public double TrimR;
            public double TrailStop;
            public double BeStop;
            public string Grade;
        }

        private struct SwingPoint
        {
            public int BarIndex;
            public double Price;
        }

        /// <summary>
        /// Parsed row from the signal CSV file.
        /// </summary>
        private class SignalRow
        {
            public int Direction;          // +1 long, -1 short
            public string Type;            // "trend" or "mss"
            public double EntryPrice;      // next bar open (pre-computed by Python)
            public double ModelStop;       // stop loss level
            public double IRL_Target;      // internal liquidity target (TP1 seed)
            public bool HasSMT;
            public string Grade;           // "A+", "B+", "C"
            public int BiasDirection;      // composite bias at signal time
            public double Regime;          // regime score
            public int SweepScore;
        }

        #endregion

        #region Parameters — Signal CSV

        [NinjaScriptProperty]
        [Display(Name = "Signal CSV Path", Order = 1, GroupName = "1. Signal CSV")]
        public string SignalCSVPath { get; set; }

        #endregion

        #region Parameters — Position Sizing

        [NinjaScriptProperty]
        [Display(Name = "Normal R ($)", Order = 1, GroupName = "2. Position")]
        public double NormalR { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Reduced R ($)", Order = 2, GroupName = "2. Position")]
        public double ReducedR { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Point Value ($)", Order = 3, GroupName = "2. Position")]
        public double PointValue { get; set; }

        #endregion

        #region Parameters — Risk

        [NinjaScriptProperty]
        [Display(Name = "Daily Max Loss R", Order = 1, GroupName = "3. Risk")]
        public double DailyMaxLossR { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max Consecutive Losses", Order = 2, GroupName = "3. Risk")]
        public int MaxConsecLosses { get; set; }

        #endregion

        #region Parameters — Trail / Trim

        [NinjaScriptProperty]
        [Display(Name = "Trim %", Order = 1, GroupName = "4. Trail")]
        public double TrimPct { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Nth Swing Trail", Order = 2, GroupName = "4. Trail")]
        public int NthSwingTrail { get; set; }

        #endregion

        #region Parameters — Swing

        [NinjaScriptProperty]
        [Display(Name = "Left Bars", Order = 1, GroupName = "5. Swing")]
        public int SwingLeftBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Right Bars", Order = 2, GroupName = "5. Swing")]
        public int SwingRightBars { get; set; }

        #endregion

        #region Parameters — Dual Mode / MSS

        [NinjaScriptProperty]
        [Display(Name = "Short RR (Trend)", Order = 1, GroupName = "6. Dual Mode")]
        public double DualShortRR { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "NY TP Multiplier", Order = 2, GroupName = "6. Dual Mode")]
        public double NYTPMult { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "MSS Long TP Mult", Order = 3, GroupName = "6. Dual Mode")]
        public double MSSLongTPMult { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "MSS Short RR", Order = 4, GroupName = "6. Dual Mode")]
        public double MSSShortRR { get; set; }

        #endregion

        #region Parameters — Regime / Misc

        [NinjaScriptProperty]
        [Display(Name = "A+ Size Mult", Order = 1, GroupName = "7. Misc")]
        public double APlusMult { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "B+ Size Mult", Order = 2, GroupName = "7. Misc")]
        public double BPlusMult { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Slippage Ticks", Order = 3, GroupName = "7. Misc")]
        public int SlippageTicks { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Commission/Side ($)", Order = 4, GroupName = "7. Misc")]
        public double CommissionPerSide { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "News CSV Path", Order = 5, GroupName = "7. Misc")]
        public string NewsCSVPath { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Min Stop ATR Mult", Order = 6, GroupName = "7. Misc")]
        public double MinStopAtrMult { get; set; }

        #endregion

        #region Parameters — Early Cut

        [NinjaScriptProperty]
        [Display(Name = "Early Cut Min Bars", Order = 1, GroupName = "8. Early Cut")]
        public int EarlyCutMinBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Early Cut Max Bars", Order = 2, GroupName = "8. Early Cut")]
        public int EarlyCutMaxBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Early Cut Wick Ratio", Order = 3, GroupName = "8. Early Cut")]
        public double EarlyCutWickRatio { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Early Cut PA Threshold", Order = 4, GroupName = "8. Early Cut")]
        public double EarlyCutPAThreshold { get; set; }

        #endregion

        #region Parameters — Live Mode

        [NinjaScriptProperty]
        [Display(Name = "Live Mode", Order = 1, GroupName = "9. Live Bridge")]
        public bool LiveMode { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Live Data Dir", Order = 2, GroupName = "9. Live Bridge")]
        public string LiveDataDir { get; set; }

        #endregion

        #region State Variables

        // Only one data series: NQ 5min (primary)
        private const int BIP_NQ5M = 0;

        // Indicators
        private ATR atrIndicator;

        // Signal map: keyed by signal bar's open time (ET)
        private Dictionary<long, SignalRow> signalMap;

        // Swing points (needed for trailing stops)
        private List<SwingPoint> swingHighs;
        private List<SwingPoint> swingLows;

        // Trade state
        private TradeState ts;
        private int barIndex;

        // Daily state
        private DateTime currentSessionDate;
        private double dailyPnlR;
        private int consecutiveLosses;
        private bool dayStopped;

        // Cumulative stats
        private double cumR;
        private double peakR;
        private int totalTrades;
        private int totalWins;
        private double totalR;

        // News blackout
        private List<DateTime> newsEventTimes;

        // Pending early cut (v8.1 Fix #10: early cut at next bar open)
        private bool pendingEarlyCut;

        // Trade log for CSV export
        private List<Dictionary<string, object>> tradeLog;

        // Debug counters
        private int dbgSignalsFound;
        private int dbgSignalsExecuted;
        private int dbgSignalsFilteredDayStop;
        private int dbgSignalsFilteredMinStop;

        // Live mode state
        private DateTime lastBarWriteTime;
        private bool liveSignalPending;
        private SignalRow liveSignalRow;

        #endregion

        #region Initialization

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Lanto NQ Strategy — CSV Signal Reader (no internal signal detection)";
                Name = "LantoNQStrategy_CSV";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = false;
                ExitOnSessionCloseSeconds = 30;
                IsFillLimitOnTouch = false;
                MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution = OrderFillResolution.Standard;
                IsInstantiatedOnEachOptimizationIteration = true;
                StartBehavior = StartBehavior.WaitUntilFlat;

                // Defaults from params.yaml / v8.1
                SignalCSVPath = "";
                NormalR = 1000;
                ReducedR = 500;
                PointValue = 2;    // MNQ micro ($2/point)
                DailyMaxLossR = 2.0;
                MaxConsecLosses = 2;
                TrimPct = 0.50;
                NthSwingTrail = 2;
                SwingLeftBars = 3;
                SwingRightBars = 1;
                DualShortRR = 0.625;
                NYTPMult = 2.0;
                MSSLongTPMult = 2.5;
                MSSShortRR = 0.50;
                APlusMult = 1.5;
                BPlusMult = 1.0;
                SlippageTicks = 1;
                CommissionPerSide = 0.62;
                NewsCSVPath = "";
                MinStopAtrMult = 1.5;
                EarlyCutMinBars = 3;
                EarlyCutMaxBars = 4;
                EarlyCutWickRatio = 0.65;
                EarlyCutPAThreshold = 0.3;

                // Live mode defaults
                LiveMode = false;
                LiveDataDir = @"C:\temp\nt8_live";
            }
            else if (State == State.Configure)
            {
                // No additional data series needed — only primary NQ 5min
            }
            else if (State == State.DataLoaded)
            {
                atrIndicator = ATR(14);

                swingHighs = new List<SwingPoint>();
                swingLows = new List<SwingPoint>();
                tradeLog = new List<Dictionary<string, object>>();
                newsEventTimes = new List<DateTime>();
                signalMap = new Dictionary<long, SignalRow>();

                ts = null;
                barIndex = 0;
                currentSessionDate = DateTime.MinValue;
                dailyPnlR = 0;
                consecutiveLosses = 0;
                dayStopped = false;
                cumR = 0;
                peakR = 0;
                totalTrades = 0;
                totalWins = 0;
                totalR = 0;
                pendingEarlyCut = false;

                if (!LiveMode)
                {
                    LoadSignalCSV();
                }
                LoadNewsCalendar();

                // Live mode initialization
                lastBarWriteTime = DateTime.MinValue;
                liveSignalPending = false;
                liveSignalRow = null;

                if (LiveMode)
                {
                    // Ensure live data directory exists
                    if (!Directory.Exists(LiveDataDir))
                        Directory.CreateDirectory(LiveDataDir);

                    Print(string.Format("=== LIVE MODE INIT === DataDir={0}", LiveDataDir));
                }
                else
                {
                    Print(string.Format("=== CSV INIT === Signals loaded: {0}, News events: {1}",
                        signalMap.Count, newsEventTimes.Count));
                }
            }
            else if (State == State.Terminated)
            {
                // Print final stats
                if (totalTrades > 0)
                {
                    double wr = (double)totalWins / totalTrades * 100.0;
                    double maxDD = peakR - cumR;
                    if (maxDD < 0) maxDD = 0;
                    double ppdd = maxDD > 0.01 ? totalR / maxDD : 0;
                    Print(string.Format("=== CSV FINAL === Trades={0} WR={1:F1}% TotalR={2:F1} MaxDD={3:F1}R PPDD={4:F1} CumR={5:F1}",
                        totalTrades, wr, totalR, maxDD, ppdd, cumR));
                    Print(string.Format("  Signals: Found={0} Executed={1} FilteredDayStop={2} FilteredMinStop={3}",
                        dbgSignalsFound, dbgSignalsExecuted, dbgSignalsFilteredDayStop, dbgSignalsFilteredMinStop));
                }

                // === CSV TRADE LOG EXPORT ===
                if (tradeLog != null && tradeLog.Count > 0)
                {
                    string csvPath = System.IO.Path.Combine(
                        Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                        "nt8_csv_trades_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".csv");
                    try
                    {
                        using (var writer = new System.IO.StreamWriter(csvPath))
                        {
                            writer.WriteLine("entry_time,exit_time,direction,signal_type,entry_price,exit_price,stop_price,tp1_price,r_multiple,exit_reason,grade,trimmed");
                            foreach (var t in tradeLog)
                            {
                                writer.WriteLine(string.Format("{0},{1},{2},{3},{4:F2},{5:F2},{6:F2},{7:F2},{8:F4},{9},{10},{11}",
                                    t["entry_time"], t["exit_time"], t["direction"], t["signal_type"],
                                    t["entry_price"], t["exit_price"], t["stop_price"], t["tp1_price"],
                                    t["r_multiple"], t["exit_reason"], t["grade"], t["trimmed"]));
                            }
                        }
                        Print("=== CSV EXPORTED === " + tradeLog.Count + " trades to " + csvPath);
                    }
                    catch (Exception ex)
                    {
                        Print("CSV export failed: " + ex.Message);
                    }
                }
            }
        }

        #endregion

        #region Signal CSV Loading

        /// <summary>
        /// Load signal CSV into signalMap keyed by signal bar's open time (ticks).
        /// Expected CSV columns (header row required):
        ///   bar_time_et, direction, type, entry_price, model_stop, irl_target,
        ///   has_smt, grade, bias_direction, regime, sweep_score
        ///
        /// bar_time_et = signal bar's OPEN time in Eastern (the bar that generated the signal).
        /// entry_price = next bar's open (pre-computed by Python).
        /// </summary>
        private void LoadSignalCSV()
        {
            if (string.IsNullOrEmpty(SignalCSVPath))
            {
                Print("WARNING: SignalCSVPath is empty — no signals will fire.");
                return;
            }

            if (!File.Exists(SignalCSVPath))
            {
                Print("ERROR: Signal CSV not found: " + SignalCSVPath);
                return;
            }

            try
            {
                var lines = File.ReadAllLines(SignalCSVPath);
                if (lines.Length < 2)
                {
                    Print("WARNING: Signal CSV has no data rows.");
                    return;
                }

                // Parse header to find column indices (flexible ordering)
                string[] header = lines[0].Split(',');
                var colIdx = new Dictionary<string, int>();
                for (int i = 0; i < header.Length; i++)
                    colIdx[header[i].Trim().ToLower()] = i;

                // Required columns
                string[] required = { "bar_time_et", "signal_dir", "signal_type", "entry_price", "model_stop" };
                foreach (var col in required)
                {
                    if (!colIdx.ContainsKey(col))
                    {
                        Print("ERROR: Signal CSV missing required column: " + col);
                        return;
                    }
                }

                int loaded = 0;
                int skipped = 0;
                for (int row = 1; row < lines.Length; row++)
                {
                    string line = lines[row].Trim();
                    if (string.IsNullOrEmpty(line)) continue;

                    string[] parts = line.Split(',');
                    if (parts.Length < required.Length) { skipped++; continue; }

                    // Parse bar_time_et (use InvariantCulture to avoid locale issues on Chinese Windows)
                    string timeStr = parts[colIdx["bar_time_et"]].Trim();
                    if (!DateTime.TryParseExact(timeStr, "yyyy-MM-dd HH:mm:ss",
                        System.Globalization.CultureInfo.InvariantCulture,
                        System.Globalization.DateTimeStyles.None, out DateTime barTimeET))
                    {
                        if (skipped < 3) Print("PARSE FAIL datetime: [" + timeStr + "]");
                        skipped++; continue;
                    }

                    // Parse direction
                    if (!int.TryParse(parts[colIdx["signal_dir"]].Trim(), out int dir))
                    { if (skipped < 3) Print("PARSE FAIL dir: [" + parts[colIdx["signal_dir"]].Trim() + "]"); skipped++; continue; }

                    // Parse entry_price (InvariantCulture for decimal point)
                    if (!double.TryParse(parts[colIdx["entry_price"]].Trim(),
                        System.Globalization.NumberStyles.Any,
                        System.Globalization.CultureInfo.InvariantCulture, out double entryPrice))
                    { if (skipped < 3) Print("PARSE FAIL entry: [" + parts[colIdx["entry_price"]].Trim() + "]"); skipped++; continue; }

                    // Parse model_stop (InvariantCulture)
                    if (!double.TryParse(parts[colIdx["model_stop"]].Trim(),
                        System.Globalization.NumberStyles.Any,
                        System.Globalization.CultureInfo.InvariantCulture, out double modelStop))
                    { if (skipped < 3) Print("PARSE FAIL stop: [" + parts[colIdx["model_stop"]].Trim() + "]"); skipped++; continue; }

                    var sig = new SignalRow
                    {
                        Direction = dir,
                        Type = GetCSVString(parts, colIdx, "signal_type", "trend"),
                        EntryPrice = entryPrice,
                        ModelStop = modelStop,
                        IRL_Target = GetCSVDouble(parts, colIdx, "irl_target", 0),
                        HasSMT = GetCSVBool(parts, colIdx, "has_smt", false),
                        Grade = GetCSVString(parts, colIdx, "grade", "B+"),
                        BiasDirection = GetCSVInt(parts, colIdx, "bias_direction", 0),
                        Regime = GetCSVDouble(parts, colIdx, "regime", 0.5),
                        SweepScore = GetCSVInt(parts, colIdx, "sweep_score", 0)
                    };

                    // Key = signal bar open time ticks
                    long key = barTimeET.Ticks;
                    if (!signalMap.ContainsKey(key))
                    {
                        signalMap[key] = sig;
                        loaded++;
                    }
                    else
                    {
                        // Duplicate time — keep the one with higher sweep score
                        if (sig.SweepScore > signalMap[key].SweepScore)
                            signalMap[key] = sig;
                        skipped++;
                    }
                }

                Print(string.Format("Signal CSV: loaded {0} signals, skipped {1} rows from {2}",
                    loaded, skipped, SignalCSVPath));
            }
            catch (Exception ex)
            {
                Print("ERROR loading signal CSV: " + ex.Message);
            }
        }

        // CSV parsing helpers
        private string GetCSVString(string[] parts, Dictionary<string, int> idx, string col, string def)
        {
            if (!idx.ContainsKey(col) || idx[col] >= parts.Length) return def;
            string val = parts[idx[col]].Trim();
            return string.IsNullOrEmpty(val) ? def : val;
        }

        private double GetCSVDouble(string[] parts, Dictionary<string, int> idx, string col, double def)
        {
            if (!idx.ContainsKey(col) || idx[col] >= parts.Length) return def;
            return double.TryParse(parts[idx[col]].Trim(), System.Globalization.NumberStyles.Any,
                System.Globalization.CultureInfo.InvariantCulture, out double val) ? val : def;
        }

        private int GetCSVInt(string[] parts, Dictionary<string, int> idx, string col, int def)
        {
            if (!idx.ContainsKey(col) || idx[col] >= parts.Length) return def;
            return int.TryParse(parts[idx[col]].Trim(), out int val) ? val : def;
        }

        private bool GetCSVBool(string[] parts, Dictionary<string, int> idx, string col, bool def)
        {
            if (!idx.ContainsKey(col) || idx[col] >= parts.Length) return def;
            string s = parts[idx[col]].Trim().ToLower();
            if (s == "true" || s == "1" || s == "yes") return true;
            if (s == "false" || s == "0" || s == "no") return false;
            return def;
        }

        #endregion

        #region Main Loop — OnBarUpdate

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != BIP_NQ5M) return;
            if (CurrentBars[BIP_NQ5M] < 50) return;

            On5mBar();
        }

        private void On5mBar()
        {
            barIndex++;
            double av = atrIndicator[0];
            if (av <= 0 || double.IsNaN(av)) return;

            // Platform timezone = Eastern (set in NT8 Options > General > Time zone).
            // Time[0] = bar CLOSE time in ET. Subtract bar period to get OPEN time.
            DateTime barOpenET = Time[0].AddMinutes(-BarsPeriod.Value);
            DateTime barTimeET = barOpenET;
            double hf = barOpenET.Hour + barOpenET.Minute / 60.0;

            // Daily reset
            NewDayCheck(barTimeET);

            // Update swings on NQ 5m (needed for trailing stops)
            UpdateSwings(High[0], Low[0], barIndex, swingHighs, swingLows,
                         Highs[BIP_NQ5M], Lows[BIP_NQ5M], CurrentBars[BIP_NQ5M]);

            // Step 1: Handle pending early cut from previous bar (v8.1 Fix #10)
            if (ts != null && pendingEarlyCut)
            {
                ExitTrade(Open[0], "early_cut_pa");
                pendingEarlyCut = false;
                // Don't return — allow new signal check below
            }

            // Step 2: Manage open position
            if (ts != null)
            {
                ManagePosition(av, hf, barTimeET);

                // Live mode: write position feedback after management
                if (LiveMode)
                    WritePositionFeedback();

                if (ts != null) return; // still in trade, skip signal lookup
            }

            // Dispatch to CSV or Live mode for signal lookup
            if (LiveMode)
                On5mBar_Live(av, hf, barTimeET, barOpenET);
            else
                On5mBar_CSV(av, hf, barTimeET, barOpenET);
        }

        /// <summary>
        /// CSV mode: look up signal from pre-loaded signal map.
        /// </summary>
        private void On5mBar_CSV(double av, double hf, DateTime barTimeET, DateTime barOpenET)
        {
            // The CSV has the signal bar's open time (bar N). Python fires signal on bar N,
            // entry is at bar N+1's open. In NT8, we are now processing bar N+1.
            // So we need to look up bar N's time = current bar's open - one bar period.
            DateTime signalBarTime = barOpenET.AddMinutes(-BarsPeriod.Value);
            long lookupKey = signalBarTime.Ticks;

            // Debug: print first 5 lookups and first 3 signal keys to diagnose matching
            if (barIndex <= 5)
            {
                Print(string.Format("DBG bar#{0} barOpenET={1} signalBarTime={2} key={3} mapSize={4}",
                    barIndex, barOpenET.ToString("yyyy-MM-dd HH:mm:ss"),
                    signalBarTime.ToString("yyyy-MM-dd HH:mm:ss"), lookupKey, signalMap.Count));
                if (barIndex == 1 && signalMap.Count > 0)
                {
                    int cnt = 0;
                    foreach (var kv in signalMap)
                    {
                        Print(string.Format("  SIG key={0} time={1} dir={2}",
                            kv.Key, new DateTime(kv.Key).ToString("yyyy-MM-dd HH:mm:ss"), kv.Value.Direction));
                        if (++cnt >= 3) break;
                    }
                }
            }

            if (signalMap.ContainsKey(lookupKey) && ts == null && !dayStopped)
            {
                var sig = signalMap[lookupKey];
                dbgSignalsFound++;

                // Basic filters (daily limits, min stop distance, news)
                if (consecutiveLosses >= MaxConsecLosses || dailyPnlR <= -DailyMaxLossR)
                {
                    dbgSignalsFilteredDayStop++;
                    return;
                }

                if (IsInNewsBlackout(barTimeET))
                    return;

                // Entry price: use the actual bar open (should closely match CSV entry_price)
                double entry = Open[0];
                double stop = sig.ModelStop;
                double stopDist = Math.Abs(entry - stop);

                // Validate stop direction
                bool stopValid = (sig.Direction == 1 && stop < entry) ||
                                 (sig.Direction == -1 && stop > entry);
                if (!stopValid) return;

                // Min stop distance filter
                if (stopDist < MinStopAtrMult * av)
                {
                    dbgSignalsFilteredMinStop++;
                    return;
                }

                // Find IRL target (use swing points for dynamic TP)
                double irl = FindIRL(sig.Direction, entry, stop);
                // If CSV provided an IRL target, prefer it; fall back to dynamic
                if (sig.IRL_Target > 0)
                    irl = sig.IRL_Target;

                // Build signal dictionary for EnterTrade (same interface as v8.1)
                var sigDict = new Dictionary<string, object>
                {
                    {"direction", sig.Direction},
                    {"type", sig.Type},
                    {"entry_price", entry},
                    {"model_stop", stop},
                    {"tp1", irl},
                    {"has_smt", sig.HasSMT},
                    {"sweep_score", sig.SweepScore},
                    {"_csv_grade", sig.Grade}  // pass grade from CSV
                };

                string session = GetSession(hf);
                EnterTrade(sigDict, av, session, hf, barTimeET);

                if (ts != null)
                    dbgSignalsExecuted++;
            }
        }

        /// <summary>
        /// Live mode: write bar data to Python, then read signal from Python on next bar.
        ///
        /// Flow per bar:
        ///   1. If there's a pending signal from Python (written on previous bar), execute it
        ///   2. Write current bar's OHLCV to current_bar.csv for Python to process
        ///   3. Write position feedback to position.csv
        /// </summary>
        private void On5mBar_Live(double av, double hf, DateTime barTimeET, DateTime barOpenET)
        {
            // Step 1: Check for pending signal from Python (written after we sent previous bar)
            if (liveSignalPending && liveSignalRow != null && ts == null && !dayStopped)
            {
                var sig = liveSignalRow;
                liveSignalPending = false;
                liveSignalRow = null;
                dbgSignalsFound++;

                // Basic filters (daily limits, news)
                if (consecutiveLosses >= MaxConsecLosses || dailyPnlR <= -DailyMaxLossR)
                {
                    dbgSignalsFilteredDayStop++;
                }
                else if (IsInNewsBlackout(barTimeET))
                {
                    // filtered by news
                }
                else
                {
                    double entry = Open[0];
                    double stop = sig.ModelStop;
                    double stopDist = Math.Abs(entry - stop);

                    bool stopValid = (sig.Direction == 1 && stop < entry) ||
                                     (sig.Direction == -1 && stop > entry);

                    if (stopValid && stopDist >= MinStopAtrMult * av)
                    {
                        double irl = sig.IRL_Target > 0 ? sig.IRL_Target : FindIRL(sig.Direction, entry, stop);

                        var sigDict = new Dictionary<string, object>
                        {
                            {"direction", sig.Direction},
                            {"type", sig.Type},
                            {"entry_price", entry},
                            {"model_stop", stop},
                            {"tp1", irl},
                            {"has_smt", sig.HasSMT},
                            {"sweep_score", sig.SweepScore},
                            {"_csv_grade", sig.Grade}
                        };

                        string session = GetSession(hf);
                        EnterTrade(sigDict, av, session, hf, barTimeET);

                        if (ts != null)
                        {
                            dbgSignalsExecuted++;
                            Print(string.Format("[LIVE] Executed signal: dir={0} type={1} entry={2:F2} stop={3:F2} grade={4}",
                                sig.Direction, sig.Type, entry, stop, sig.Grade));
                        }
                    }
                    else if (!stopValid)
                    {
                        Print(string.Format("[LIVE] Signal rejected: invalid stop direction dir={0} entry={1:F2} stop={2:F2}",
                            sig.Direction, entry, stop));
                    }
                    else
                    {
                        dbgSignalsFilteredMinStop++;
                        Print(string.Format("[LIVE] Signal rejected: stop too small {0:F2} < {1:F2}",
                            stopDist, MinStopAtrMult * av));
                    }
                }
            }

            // Step 2: Read signal.csv from Python (from previous bar's computation)
            ReadLiveSignal();

            // Step 3: Write current bar data for Python to process
            WriteLiveBar(barOpenET);

            // Step 4: Write position feedback
            WritePositionFeedback();
        }

        /// <summary>
        /// Write current bar OHLCV to current_bar.csv for Python signal server.
        /// Uses atomic write (write to .tmp then rename) to prevent partial reads.
        /// </summary>
        private void WriteLiveBar(DateTime barOpenET)
        {
            string filePath = Path.Combine(LiveDataDir, "current_bar.csv");
            string tmpPath = filePath + ".tmp";

            try
            {
                string content = string.Format(
                    "time_et,open,high,low,close,volume\n{0},{1},{2},{3},{4},{5}\n",
                    barOpenET.ToString("yyyy-MM-dd HH:mm:ss"),
                    Open[0].ToString(System.Globalization.CultureInfo.InvariantCulture),
                    High[0].ToString(System.Globalization.CultureInfo.InvariantCulture),
                    Low[0].ToString(System.Globalization.CultureInfo.InvariantCulture),
                    Close[0].ToString(System.Globalization.CultureInfo.InvariantCulture),
                    (long)Volume[0]);

                File.WriteAllText(tmpPath, content);
                if (File.Exists(filePath))
                    File.Delete(filePath);
                File.Move(tmpPath, filePath);

                lastBarWriteTime = DateTime.Now;
            }
            catch (Exception ex)
            {
                Print("[LIVE] Failed to write bar: " + ex.Message);
            }
        }

        /// <summary>
        /// Read signal.csv from Python signal server.
        /// If signal found, store as pending for execution on next bar.
        /// </summary>
        private void ReadLiveSignal()
        {
            string filePath = Path.Combine(LiveDataDir, "signal.csv");
            if (!File.Exists(filePath)) return;

            try
            {
                string[] lines = File.ReadAllLines(filePath);
                if (lines.Length < 2) return;

                string[] header = lines[0].Split(',');
                string[] values = lines[1].Split(',');

                var colIdx = new Dictionary<string, int>();
                for (int i = 0; i < header.Length; i++)
                    colIdx[header[i].Trim().ToLower()] = i;

                // Check has_signal
                string hasSignal = GetCSVString(values, colIdx, "has_signal", "false");
                if (hasSignal.ToLower() != "true")
                {
                    // No signal — clean up
                    try { File.Delete(filePath); } catch { }
                    return;
                }

                // Parse signal
                int dir = GetCSVInt(values, colIdx, "direction", 0);
                if (dir == 0) return;

                var sig = new SignalRow
                {
                    Direction = dir,
                    Type = GetCSVString(values, colIdx, "type", "trend"),
                    EntryPrice = GetCSVDouble(values, colIdx, "entry_price", 0),
                    ModelStop = GetCSVDouble(values, colIdx, "stop", 0),
                    IRL_Target = GetCSVDouble(values, colIdx, "tp1", 0),
                    HasSMT = GetCSVBool(values, colIdx, "has_smt", false),
                    Grade = GetCSVString(values, colIdx, "grade", "B+"),
                    BiasDirection = GetCSVInt(values, colIdx, "bias_direction", 0),
                    Regime = GetCSVDouble(values, colIdx, "regime", 0.5),
                    SweepScore = GetCSVInt(values, colIdx, "sweep_score", 0)
                };

                liveSignalRow = sig;
                liveSignalPending = true;

                Print(string.Format("[LIVE] Signal received: dir={0} type={1} entry={2:F2} stop={3:F2} tp1={4:F2} grade={5}",
                    sig.Direction, sig.Type, sig.EntryPrice, sig.ModelStop, sig.IRL_Target, sig.Grade));

                // Delete signal file to acknowledge
                try { File.Delete(filePath); } catch { }
            }
            catch (Exception ex)
            {
                Print("[LIVE] Failed to read signal: " + ex.Message);
            }
        }

        /// <summary>
        /// Write position feedback to position.csv so Python can track state.
        /// </summary>
        private void WritePositionFeedback()
        {
            string filePath = Path.Combine(LiveDataDir, "position.csv");
            string tmpPath = filePath + ".tmp";

            try
            {
                bool inPos = ts != null;
                int dir = inPos ? ts.Direction : 0;
                double entryP = inPos ? ts.EntryPrice : 0;
                double stopP = inPos ? ts.StopPrice : 0;
                bool trimmed = inPos ? ts.Trimmed : false;

                string content = string.Format(
                    "in_position,direction,entry_price,stop,trimmed,daily_pnl_r,consecutive_losses\n" +
                    "{0},{1},{2},{3},{4},{5},{6}\n",
                    inPos.ToString().ToLower(),
                    dir,
                    entryP.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    stopP.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    trimmed.ToString().ToLower(),
                    dailyPnlR.ToString("F4", System.Globalization.CultureInfo.InvariantCulture),
                    consecutiveLosses);

                File.WriteAllText(tmpPath, content);
                if (File.Exists(filePath))
                    File.Delete(filePath);
                File.Move(tmpPath, filePath);
            }
            catch (Exception ex)
            {
                Print("[LIVE] Failed to write position: " + ex.Message);
            }
        }

        #endregion

        #region Trade Management

        private void EnterTrade(Dictionary<string, object> sig, double av,
            string session, double hf, DateTime barTimeET)
        {
            int dir = (int)sig["direction"];
            double entry = (double)sig["entry_price"];
            double stop = (double)sig["model_stop"];
            double tp1 = (double)sig["tp1"];
            string sigType = (string)sig["type"];

            // Apply slippage
            double slippagePoints = SlippageTicks * 0.25;
            double actualEntry = dir == 1 ? entry + slippagePoints : entry - slippagePoints;
            double stopDist = Math.Abs(actualEntry - stop);
            if (stopDist < 1.0) return;

            // Grade: use CSV-provided grade if available, otherwise default to B+
            string grade = sig.ContainsKey("_csv_grade") ? (string)sig["_csv_grade"] : "B+";

            // Position sizing
            bool isReduced = barTimeET.DayOfWeek == DayOfWeek.Monday ||
                             barTimeET.DayOfWeek == DayOfWeek.Friday;
            double baseR = isReduced ? ReducedR : NormalR;
            double rAmount;
            if (grade == "A+") rAmount = baseR * APlusMult;
            else if (grade == "B+") rAmount = baseR * BPlusMult;
            else rAmount = baseR * 0.5;

            if (rAmount <= 0) return;

            int contracts = Math.Max(1, (int)(rAmount / (stopDist * PointValue)));

            // TP adjustment (same logic as v8.1)
            bool isMSS = sigType == "mss";
            if (dir == 1) // Long
            {
                double tpDist = Math.Abs(tp1 - actualEntry);
                double tpMult = isMSS ? MSSLongTPMult : NYTPMult;
                if (hf >= 9.5 && hf < 16.0)
                    tp1 = actualEntry + tpDist * tpMult;
            }
            else // Short — fixed R:R scalp
            {
                double shortRR = isMSS ? MSSShortRR : DualShortRR;
                tp1 = actualEntry - stopDist * shortRR;
            }

            // Submit order (managed mode)
            if (dir == 1)
                EnterLong(contracts, "LantoLong");
            else
                EnterShort(contracts, "LantoShort");

            ts = new TradeState
            {
                Direction = dir,
                EntryPrice = actualEntry,
                StopPrice = stop,
                TP1Price = tp1,
                Contracts = contracts,
                OrigContracts = contracts,
                EntryTime = Time[0],
                EntryBarIdx = barIndex,
                SignalType = sigType,
                OrigStopDist = stopDist,
                Grade = grade,
                TrailStop = 0,
                BeStop = 0
            };
        }

        private void ManagePosition(double av, double hf, DateTime barTimeET)
        {
            if (ts == null) return;
            int barsInTrade = barIndex - ts.EntryBarIdx;

            // --- Early PA cut --- (Fix #10: exit at next bar's open)
            if (!ts.Trimmed && barsInTrade >= EarlyCutMinBars && barsInTrade <= EarlyCutMaxBars)
            {
                if (CheckEarlyCut(av))
                {
                    pendingEarlyCut = true;
                    return;
                }
            }

            if (ts.Direction == 1) // LONG
            {
                // Effective stop
                double effectiveStop = ts.StopPrice;
                if (ts.Trimmed)
                {
                    effectiveStop = ts.TrailStop > 0 ? ts.TrailStop : ts.StopPrice;
                    if (ts.BeStop > 0 && ts.BeStop > effectiveStop)
                        effectiveStop = ts.BeStop;
                }

                // Stop hit
                if (Low[0] <= effectiveStop)
                {
                    double exitP = effectiveStop - SlippageTicks * 0.25;
                    string reason = (ts.Trimmed && effectiveStop >= ts.EntryPrice) ? "be_sweep" : "stop";
                    ExitTrade(exitP, reason);
                    return;
                }

                // TP1 hit (before trim)
                if (!ts.Trimmed && High[0] >= ts.TP1Price)
                {
                    TrimPosition();
                    if (ts == null) return; // fully exited if 100% trim
                }

                // Update trailing stop
                if (ts != null && ts.Trimmed)
                {
                    double newTrail = FindNthSwing(swingLows, NthSwingTrail, 1, ts.EntryBarIdx, ts.EntryPrice);
                    if (!double.IsNaN(newTrail) && newTrail > ts.TrailStop)
                        ts.TrailStop = newTrail;
                }
            }
            else // SHORT
            {
                double effectiveStop = ts.StopPrice;
                if (ts.Trimmed)
                {
                    effectiveStop = ts.TrailStop > 0 ? ts.TrailStop : ts.StopPrice;
                    if (ts.BeStop > 0 && ts.BeStop < effectiveStop)
                        effectiveStop = ts.BeStop;
                }

                if (High[0] >= effectiveStop)
                {
                    double exitP = effectiveStop + SlippageTicks * 0.25;
                    string reason = (ts.Trimmed && effectiveStop <= ts.EntryPrice) ? "be_sweep" : "stop";
                    ExitTrade(exitP, reason);
                    return;
                }

                if (!ts.Trimmed && Low[0] <= ts.TP1Price)
                {
                    TrimPosition();
                    if (ts == null) return;
                }

                if (ts != null && ts.Trimmed)
                {
                    double newTrail = FindNthSwing(swingHighs, NthSwingTrail, -1, ts.EntryBarIdx, ts.EntryPrice);
                    if (!double.IsNaN(newTrail) && newTrail < ts.TrailStop)
                        ts.TrailStop = newTrail;
                }
            }
        }

        private void TrimPosition()
        {
            int trimQty;
            bool fullExit;

            if (ts.Direction == -1) // shorts: always 100% exit
            {
                trimQty = ts.Contracts;
                fullExit = true;
            }
            else
            {
                trimQty = Math.Max(1, (int)(ts.OrigContracts * TrimPct));
                fullExit = (ts.Contracts - trimQty) <= 0;
            }

            // Calculate trim R
            double tpDist = Math.Abs(ts.TP1Price - ts.EntryPrice);
            double trimRatio = (double)trimQty / ts.OrigContracts;
            ts.TrimR = (ts.OrigStopDist > 0) ? (tpDist / ts.OrigStopDist) * trimRatio : 0;

            if (fullExit)
            {
                ExitTrade(ts.TP1Price, "tp1");
                return;
            }

            // Partial trim
            if (ts.Direction == 1)
                ExitLong(trimQty, "TrimLong", "LantoLong");
            else
                ExitShort(trimQty, "TrimShort", "LantoShort");

            ts.Contracts -= trimQty;
            ts.Trimmed = true;
            ts.BeStop = ts.EntryPrice;

            // Initialize trail stop
            if (ts.Direction == 1)
            {
                ts.TrailStop = FindNthSwing(swingLows, NthSwingTrail, 1, ts.EntryBarIdx, ts.EntryPrice);
                if (double.IsNaN(ts.TrailStop) || ts.TrailStop <= 0) ts.TrailStop = ts.BeStop;
            }
            else
            {
                ts.TrailStop = FindNthSwing(swingHighs, NthSwingTrail, -1, ts.EntryBarIdx, ts.EntryPrice);
                if (double.IsNaN(ts.TrailStop) || ts.TrailStop <= 0) ts.TrailStop = ts.BeStop;
            }
        }

        private bool CheckEarlyCut(double av)
        {
            int n = Math.Min(barIndex - ts.EntryBarIdx, CurrentBars[BIP_NQ5M]);
            if (n < EarlyCutMinBars) return false;

            double sumWick = 0;
            int favCount = 0;
            int validBars = 0;

            for (int j = 0; j < n; j++)
            {
                double range = High[j] - Low[j];
                if (range <= 0) continue;
                double body = Math.Abs(Close[j] - Open[j]);
                double wick = 1.0 - body / range;
                sumWick += wick;
                if ((ts.Direction == 1 && Close[j] > Open[j]) ||
                    (ts.Direction == -1 && Close[j] < Open[j]))
                    favCount++;
                validBars++;
            }

            if (validBars < 2) return false;
            double avgWick = sumWick / validBars;
            double favorable = (double)favCount / validBars;

            // Price displacement from entry
            double disp = ts.Direction == 1 ? Close[0] - ts.EntryPrice : ts.EntryPrice - Close[0];
            bool noProgress = disp < av * EarlyCutPAThreshold;

            return avgWick > EarlyCutWickRatio && favorable < 0.5 && noProgress;
        }

        private void ExitTrade(double exitPrice, string reason)
        {
            if (ts == null) return;

            // Close all remaining
            if (ts.Direction == 1)
                ExitLong(ts.Contracts, "ExitLong", "LantoLong");
            else
                ExitShort(ts.Contracts, "ExitShort", "LantoShort");

            // PnL calculation
            double pp = ts.Direction == 1 ? exitPrice - ts.EntryPrice : ts.EntryPrice - exitPrice;
            double rm;

            if (ts.Trimmed && reason != "tp1")
            {
                double remainRatio = (double)ts.Contracts / ts.OrigContracts;
                double remainR = ts.OrigStopDist > 0 ? (pp / ts.OrigStopDist) * remainRatio : 0;
                rm = ts.TrimR + remainR;
                if (reason == "stop" && Math.Abs(pp) < 1) reason = "be_sweep";
            }
            else if (reason == "tp1")
            {
                double tpDist = Math.Abs(ts.TP1Price - ts.EntryPrice);
                rm = ts.OrigStopDist > 0 ? tpDist / ts.OrigStopDist : 0;
            }
            else
            {
                rm = ts.OrigStopDist > 0 ? pp / ts.OrigStopDist : 0;
            }

            cumR += rm;
            if (cumR > peakR) peakR = cumR;
            dailyPnlR += rm;
            totalR += rm;
            totalTrades++;

            if (rm > 0) totalWins++;

            // Loss tracking for 0-for-2
            if (reason == "be_sweep" && ts.Trimmed)
            {
                // BE sweep after trim is profitable — NOT a loss
            }
            else if (rm < 0)
            {
                consecutiveLosses++;
            }
            else
            {
                consecutiveLosses = 0;
            }

            // Check daily limits
            if (consecutiveLosses >= MaxConsecLosses) dayStopped = true;
            if (dailyPnlR <= -DailyMaxLossR) dayStopped = true;

            Print(string.Format("[{0}] d={1} type={2} e={3:F1} x={4:F1} R={5:F2} cum={6:F1} grade={7}",
                reason, ts.Direction, ts.SignalType, ts.EntryPrice, exitPrice, rm, cumR, ts.Grade));

            // Log trade for CSV export
            if (tradeLog != null)
            {
                tradeLog.Add(new Dictionary<string, object>
                {
                    {"entry_time", ts.EntryTime.ToString("yyyy-MM-dd HH:mm:ss")},
                    {"exit_time", Time[0].ToString("yyyy-MM-dd HH:mm:ss")},
                    {"direction", ts.Direction},
                    {"signal_type", ts.SignalType},
                    {"entry_price", ts.EntryPrice},
                    {"exit_price", exitPrice},
                    {"stop_price", ts.StopPrice},
                    {"tp1_price", ts.TP1Price},
                    {"r_multiple", rm},
                    {"exit_reason", reason},
                    {"grade", ts.Grade},
                    {"trimmed", ts.Trimmed}
                });
            }

            ts = null;
        }

        #endregion

        #region Utilities

        private string GetSession(double hf)
        {
            if (hf >= 9.5 && hf < 16.0) return "ny";
            if (hf >= 3.0 && hf < 9.5) return "london";
            return "asia";
        }

        private void NewDayCheck(DateTime barTimeET)
        {
            DateTime sessionDate = barTimeET.Date;
            if (barTimeET.Hour >= 18)
                sessionDate = barTimeET.Date.AddDays(1);

            if (sessionDate != currentSessionDate)
            {
                currentSessionDate = sessionDate;
                dailyPnlR = 0;
                consecutiveLosses = 0;
                dayStopped = false;
                pendingEarlyCut = false;
            }
        }

        private void UpdateSwings(double high, double low, int idx,
            List<SwingPoint> shList, List<SwingPoint> slList,
            ISeries<double> highSeries, ISeries<double> lowSeries, int currentBar)
        {
            int right = SwingRightBars;
            int left = SwingLeftBars;
            if (currentBar < left + right + 1) return;

            double candidateHigh = highSeries[right];
            double candidateLow = lowSeries[right];

            // Check swing high
            bool isSwingHigh = true;
            for (int j = right + 1; j <= right + left; j++)
            {
                if (highSeries[j] >= candidateHigh) { isSwingHigh = false; break; }
            }
            if (isSwingHigh)
            {
                for (int j = 0; j < right; j++)
                {
                    if (highSeries[j] >= candidateHigh) { isSwingHigh = false; break; }
                }
            }

            // Check swing low
            bool isSwingLow = true;
            for (int j = right + 1; j <= right + left; j++)
            {
                if (lowSeries[j] <= candidateLow) { isSwingLow = false; break; }
            }
            if (isSwingLow)
            {
                for (int j = 0; j < right; j++)
                {
                    if (lowSeries[j] <= candidateLow) { isSwingLow = false; break; }
                }
            }

            if (isSwingHigh)
            {
                shList.Add(new SwingPoint { BarIndex = idx - right, Price = candidateHigh });
                if (shList.Count > 50) shList.RemoveAt(0);
            }
            if (isSwingLow)
            {
                slList.Add(new SwingPoint { BarIndex = idx - right, Price = candidateLow });
                if (slList.Count > 50) slList.RemoveAt(0);
            }
        }

        /// <summary>
        /// Find nth most recent swing (no filtering). Matches Python / v8.1 behavior.
        /// </summary>
        private double FindNthSwing(List<SwingPoint> swings, int n, int dir,
            int entryBarIdx, double entryPrice)
        {
            int count = 0;
            for (int j = swings.Count - 1; j >= 0; j--)
            {
                var s = swings[j];
                count++;
                if (count == n) return s.Price;
            }
            return double.NaN;
        }

        /// <summary>
        /// Find Internal Liquidity (IRL) target — nearest swing beyond entry.
        /// Used as fallback if CSV does not provide irl_target.
        /// </summary>
        private double FindIRL(int dir, double entry, double stop)
        {
            double risk = Math.Abs(entry - stop);
            if (dir == 1)
            {
                double best = double.MaxValue;
                foreach (var sh in swingHighs)
                {
                    if (sh.Price > entry && sh.Price < best) best = sh.Price;
                }
                return best < double.MaxValue ? best : entry + risk * 2.0;
            }
            else
            {
                double best = double.MinValue;
                foreach (var sl in swingLows)
                {
                    if (sl.Price < entry && sl.Price > best) best = sl.Price;
                }
                return best > double.MinValue ? best : entry - risk * 2.0;
            }
        }

        private bool IsInNewsBlackout(DateTime barTimeET)
        {
            if (newsEventTimes == null || newsEventTimes.Count == 0) return false;

            foreach (var evt in newsEventTimes)
            {
                double minBefore = (evt - barTimeET).TotalMinutes;
                double minAfter = (barTimeET - evt).TotalMinutes;

                if (minBefore >= 0 && minBefore <= 60) return true;
                if (minAfter >= 0 && minAfter <= 5) return true;
            }
            return false;
        }

        private void LoadNewsCalendar()
        {
            if (string.IsNullOrEmpty(NewsCSVPath) || !File.Exists(NewsCSVPath))
                return;

            try
            {
                var lines = File.ReadAllLines(NewsCSVPath);
                foreach (var line in lines.Skip(1))
                {
                    var parts = line.Split(',');
                    if (parts.Length < 4) continue;

                    string impact = parts[3].Trim().ToLower();
                    if (impact != "high") continue;

                    string dateStr = parts[0].Trim();
                    string timeStr = parts[1].Trim();

                    if (DateTime.TryParse(dateStr + " " + timeStr, out DateTime eventTime))
                    {
                        newsEventTimes.Add(eventTime);
                    }
                }
                Print(string.Format("Loaded {0} high-impact news events", newsEventTimes.Count));
            }
            catch (Exception ex)
            {
                Print("Failed to load news calendar: " + ex.Message);
            }
        }

        #endregion
    }
}
