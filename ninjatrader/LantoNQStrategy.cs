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
    /// Lanto NQ Quantitative Strategy — NinjaTrader 8 port.
    /// VERSION: v8.1 (2026-04-02)
    /// v8.1: 4 critical audit fixes — HTF FVG invalidation, TP1 recalc at execution,
    ///       deferred FVG "used" marking, overnight bias uses NY open price
    /// v8: 11-fix alignment pass — FVG state update order (#7), entry at next bar open (#3),
    ///     fluency max(bull,bear)/window (#2), full HTF FVG bias (#4), regime PDA count (#5),
    ///     displacement engulfment (#6), IRL dynamic fallback (#8), trail swing unfiltered (#9),
    ///     early cut next bar open (#10), rollover filter (#1), remove DD sizing
    /// Python reference: 534 trades, +156.63R, WR=46.1%
    /// </summary>
    public class LantoNQStrategy : Strategy
    {
        #region Data Classes

        private class FVG
        {
            public int BarIndex;
            public int Direction;       // +1 bull, -1 bear
            public double Top;
            public double Bottom;
            public double Size;
            public double Candle2Open;  // model stop level
            public string Status;       // untested, tested_rejected, invalidated
            public DateTime CreationTime;
            public int LastSignalIdx;
            public bool IsIFVG;
            public int IFVGDirection;   // direction after inversion
            public bool SweptLiquidity;
            public int SweepScore;

            public FVG()
            {
                Status = "untested";
                LastSignalIdx = -999;
            }
        }

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

        #endregion

        #region Parameters — Displacement

        [NinjaScriptProperty]
        [Display(Name = "ATR Mult", Order = 1, GroupName = "1. Displacement")]
        public double DispAtrMult { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Body Ratio", Order = 2, GroupName = "1. Displacement")]
        public double DispBodyRatio { get; set; }

        #endregion

        #region Parameters — Fluency

        [NinjaScriptProperty]
        [Display(Name = "Window", Order = 1, GroupName = "2. Fluency")]
        public int FluencyWindow { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Threshold", Order = 2, GroupName = "2. Fluency")]
        public double FluencyThreshold { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "W Directional", Order = 3, GroupName = "2. Fluency")]
        public double FluencyWDir { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "W Body Ratio", Order = 4, GroupName = "2. Fluency")]
        public double FluencyWBody { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "W Bar Size", Order = 5, GroupName = "2. Fluency")]
        public double FluencyWBar { get; set; }

        #endregion

        #region Parameters — Swing

        [NinjaScriptProperty]
        [Display(Name = "Left Bars", Order = 1, GroupName = "3. Swing")]
        public int SwingLeftBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Right Bars", Order = 2, GroupName = "3. Swing")]
        public int SwingRightBars { get; set; }

        #endregion

        #region Parameters — FVG

        [NinjaScriptProperty]
        [Display(Name = "Min Size ATR Mult", Order = 1, GroupName = "4. FVG")]
        public double FVGMinSizeAtr { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Signal Cooldown Bars", Order = 2, GroupName = "4. FVG")]
        public int FVGCooldown { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Rejection Body Ratio", Order = 3, GroupName = "4. FVG")]
        public double FVGRejectionBodyRatio { get; set; }

        #endregion

        #region Parameters — Position Sizing

        [NinjaScriptProperty]
        [Display(Name = "Normal R ($)", Order = 1, GroupName = "5. Position")]
        public double NormalR { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Reduced R ($)", Order = 2, GroupName = "5. Position")]
        public double ReducedR { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Point Value ($)", Order = 3, GroupName = "5. Position")]
        public double PointValue { get; set; }

        #endregion

        #region Parameters — Risk

        [NinjaScriptProperty]
        [Display(Name = "Daily Max Loss R", Order = 1, GroupName = "6. Risk")]
        public double DailyMaxLossR { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max Consecutive Losses", Order = 2, GroupName = "6. Risk")]
        public int MaxConsecLosses { get; set; }

        #endregion

        #region Parameters — Sessions

        [NinjaScriptProperty]
        [Display(Name = "Skip London (Trend)", Order = 1, GroupName = "7. Sessions")]
        public bool SkipLondon { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Skip Asia", Order = 2, GroupName = "7. Sessions")]
        public bool SkipAsia { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "ORM Window (min)", Order = 3, GroupName = "7. Sessions")]
        public int ORMWindowMin { get; set; }

        #endregion

        #region Parameters — Signal Quality

        [NinjaScriptProperty]
        [Display(Name = "Enabled", Order = 1, GroupName = "8. Signal Quality")]
        public bool SQEnabled { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Long Threshold", Order = 2, GroupName = "8. Signal Quality")]
        public double SQLongThreshold { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Short Threshold", Order = 3, GroupName = "8. Signal Quality")]
        public double SQShortThreshold { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "W Size", Order = 4, GroupName = "8. Signal Quality")]
        public double SQWSize { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "W Disp", Order = 5, GroupName = "8. Signal Quality")]
        public double SQWDisp { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "W Flu", Order = 6, GroupName = "8. Signal Quality")]
        public double SQWFlu { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "W PA", Order = 7, GroupName = "8. Signal Quality")]
        public double SQWPA { get; set; }

        #endregion

        #region Parameters — Dual Mode

        [NinjaScriptProperty]
        [Display(Name = "Short RR (Trend)", Order = 1, GroupName = "9. Dual Mode")]
        public double DualShortRR { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "NY TP Multiplier", Order = 2, GroupName = "9. Dual Mode")]
        public double NYTPMult { get; set; }

        #endregion

        #region Parameters — MSS Management

        [NinjaScriptProperty]
        [Display(Name = "MSS Long TP Mult", Order = 1, GroupName = "10. MSS")]
        public double MSSLongTPMult { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "MSS Short RR", Order = 2, GroupName = "10. MSS")]
        public double MSSShortRR { get; set; }

        #endregion

        #region Parameters — SMT

        [NinjaScriptProperty]
        [Display(Name = "SMT Enabled", Order = 1, GroupName = "11. SMT")]
        public bool SMTEnabled { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Sweep Lookback", Order = 2, GroupName = "11. SMT")]
        public int SMTSweepLookback { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Time Tolerance", Order = 3, GroupName = "11. SMT")]
        public int SMTTimeTolerance { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Require For MSS", Order = 4, GroupName = "11. SMT")]
        public bool SMTRequireForMSS { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Bypass Session Filter", Order = 5, GroupName = "11. SMT")]
        public bool SMTBypassSession { get; set; }

        #endregion

        #region Parameters — Trail / Trim

        [NinjaScriptProperty]
        [Display(Name = "Trim %", Order = 1, GroupName = "12. Trail")]
        public double TrimPct { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Nth Swing Trail", Order = 2, GroupName = "12. Trail")]
        public int NthSwingTrail { get; set; }

        #endregion

        #region Parameters — Regime / Misc

        [NinjaScriptProperty]
        [Display(Name = "Min Stop ATR Mult", Order = 1, GroupName = "13. Regime")]
        public double MinStopAtrMult { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Lunch Skip Start (ET frac)", Order = 2, GroupName = "13. Regime")]
        public double LunchStart { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Lunch Skip End (ET frac)", Order = 3, GroupName = "13. Regime")]
        public double LunchEnd { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "A+ Size Mult", Order = 4, GroupName = "13. Regime")]
        public double APlusMult { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "B+ Size Mult", Order = 5, GroupName = "13. Regime")]
        public double BPlusMult { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Slippage Ticks", Order = 6, GroupName = "13. Regime")]
        public int SlippageTicks { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Commission/Side ($)", Order = 7, GroupName = "13. Regime")]
        public double CommissionPerSide { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "ES Instrument", Order = 8, GroupName = "13. Regime")]
        public string ESInstrument { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "News CSV Path", Order = 9, GroupName = "13. Regime")]
        public string NewsCSVPath { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Early Cut Min Bars", Order = 10, GroupName = "13. Regime")]
        public int EarlyCutMinBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Early Cut Max Bars", Order = 11, GroupName = "13. Regime")]
        public int EarlyCutMaxBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Early Cut Wick Ratio", Order = 12, GroupName = "13. Regime")]
        public double EarlyCutWickRatio { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Early Cut PA Threshold", Order = 13, GroupName = "13. Regime")]
        public double EarlyCutPAThreshold { get; set; }

        #endregion

        #region State Variables

        // Data series indices (BarsInProgress)
        private const int BIP_NQ5M = 0;   // Primary: NQ 5min
        private const int BIP_NQ1H = 1;   // NQ 1H for bias (resample from 5m)
        private const int BIP_NQ4H = 2;   // NQ 4H for bias (resample from 5m)
        private const int BIP_ES5M = 3;   // ES 5min for SMT

        // Indicators
        private ATR atrIndicator;

        // FVG pools
        private List<FVG> fvgs5m;
        private List<FVG> fvgs1h;
        private List<FVG> fvgs4h;

        // Swing points
        private List<SwingPoint> swingHighs;
        private List<SwingPoint> swingLows;
        // ES swings for SMT
        private List<SwingPoint> esSwingHighs;
        private List<SwingPoint> esSwingLows;

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

        // Bias
        private int bias;
        private double lastHTFBias;
        private int htfPDACount;
        private double overnightHigh;
        private double overnightLow;
        private double ormHigh;
        private double ormLow;
        private bool ormLocked;
        private int ormBias;

        // Session liquidity tracking (Python _compute_liquidity_levels)
        // 5 sub-sessions: Asia(18-3), London(3-9:30), NY_AM(9:30-11), NY_Lunch(11-13), NY_PM(13-16)
        // Only COMPLETED session H/L exposed (not current running)
        private double[] sessRunningH;  // [5] running highs
        private double[] sessRunningL;  // [5] running lows
        private double[] sessCompletedH; // [5] completed (frozen) highs
        private double[] sessCompletedL; // [5] completed (frozen) lows
        private bool[] sessActive;       // [5] currently in this session?

        // HTF swings for sweep scoring (left=10, right=3, separate from 5m swings)
        private List<SwingPoint> htfSwingHighs;
        private List<SwingPoint> htfSwingLows;

        // News blackout
        private List<DateTime> newsEventTimes;

        // Timezone: platform must be set to Eastern (NT8 Options > General > Time zone)
        // No runtime timezone conversion needed — Time[0] is already ET.

        // Debug counters
        private int dbgNYBars, dbgLondonBars, dbgAsiaBars;
        private int dbgSignalsNY, dbgSignalsLondon, dbgSignalsAsia;
        private int dbgFilteredSession, dbgFilteredBias, dbgFilteredSQ, dbgFilteredMinStop;

        // Trade log for CSV export
        private List<Dictionary<string, object>> tradeLog;

        // SMT state
        private double lastNQSwingHighPrice;
        private double lastNQSwingLowPrice;
        private double lastESSwingHighPrice;
        private double lastESSwingLowPrice;
        private bool currentSMTBull;
        private bool currentSMTBear;

        // Rolling buffers for ES
        private List<double> esHighBuffer;
        private List<double> esLowBuffer;

        // Pending signal mechanism (Fix #3: entry at next bar open)
        private Dictionary<string, object> pendingSignal;

        // Pending early cut (Fix #10: early cut at next bar open)
        private bool pendingEarlyCut;

        // NY open price for overnight bias (v8.1: use locked price, not live Close[0])
        private double nyOpenPrice;

        // Rollover dates (Fix #1: discard FVGs spanning roll dates)
        private HashSet<DateTime> rollDates;

        #endregion

        #region Initialization

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Lanto NQ ICT Strategy — FVG + SMT + Dual Mode";
                Name = "LantoNQStrategy";
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

                // Defaults from params.yaml
                DispAtrMult = 0.8;
                DispBodyRatio = 0.60;
                FluencyWindow = 6;
                FluencyThreshold = 0.60;
                FluencyWDir = 0.4;
                FluencyWBody = 0.3;
                FluencyWBar = 0.3;
                SwingLeftBars = 3;
                SwingRightBars = 1;
                FVGMinSizeAtr = 0.5;
                FVGCooldown = 10;
                FVGRejectionBodyRatio = 0.55;
                NormalR = 1000;
                ReducedR = 500;
                PointValue = 2;    // MNQ micro ($2/point) — matches params.yaml point_value: 2
                DailyMaxLossR = 2.0;
                MaxConsecLosses = 2;
                SkipLondon = true;
                SkipAsia = true;
                ORMWindowMin = 30;
                SQEnabled = true;
                SQLongThreshold = 0.68;
                SQShortThreshold = 0.82;
                SQWSize = 0.30;
                SQWDisp = 0.30;
                SQWFlu = 0.20;
                SQWPA = 0.20;
                DualShortRR = 0.625;
                NYTPMult = 2.0;
                MSSLongTPMult = 2.5;
                MSSShortRR = 0.50;
                SMTEnabled = true;
                SMTSweepLookback = 15;
                SMTTimeTolerance = 1;
                SMTRequireForMSS = true;
                SMTBypassSession = true;
                TrimPct = 0.50;
                NthSwingTrail = 2;
                MinStopAtrMult = 1.7;
                LunchStart = 12.5;
                LunchEnd = 13.0;
                APlusMult = 1.5;
                BPlusMult = 1.0;
                SlippageTicks = 1;
                CommissionPerSide = 0.62;
                ESInstrument = "ES_5M_NT8";
                NewsCSVPath = "";
                EarlyCutMinBars = 3;
                EarlyCutMaxBars = 4;
                EarlyCutWickRatio = 0.65;
                EarlyCutPAThreshold = 0.3;
            }
            else if (State == State.Configure)
            {
                AddDataSeries(BarsPeriodType.Minute, 60);    // BIP 1: NQ 1H (resample from 5m)
                AddDataSeries(BarsPeriodType.Minute, 240);   // BIP 2: NQ 4H (resample from 5m)

                // ES 5m for SMT divergence — hardcoded instrument name (NT8 requirement)
                if (SMTEnabled)
                    AddDataSeries("ES_5M_NT8", BarsPeriodType.Minute, 5); // BIP 3
            }
            else if (State == State.DataLoaded)
            {
                // Diagnostic: verify data series loaded
                Print(string.Format("=== v8 INIT === BarsArray.Length={0} SMTEnabled={1} BIP_ES5M={2}",
                    BarsArray.Length, SMTEnabled, BIP_ES5M));
                for (int b = 0; b < BarsArray.Length; b++)
                    Print(string.Format("  BIP {0}: {1} {2}min", b,
                        BarsArray[b].Instrument.FullName, BarsArray[b].BarsPeriod.Value));

                atrIndicator = ATR(14);

                fvgs5m = new List<FVG>();
                fvgs1h = new List<FVG>();
                fvgs4h = new List<FVG>();
                swingHighs = new List<SwingPoint>();
                swingLows = new List<SwingPoint>();
                esSwingHighs = new List<SwingPoint>();
                esSwingLows = new List<SwingPoint>();
                esHighBuffer = new List<double>();
                esLowBuffer = new List<double>();

                tradeLog = new List<Dictionary<string, object>>();

                // Session liquidity
                sessRunningH = new double[5];
                sessRunningL = new double[5];
                sessCompletedH = new double[5];
                sessCompletedL = new double[5];
                sessActive = new bool[5];
                for (int s = 0; s < 5; s++)
                {
                    sessRunningH[s] = double.NaN;
                    sessRunningL[s] = double.NaN;
                    sessCompletedH[s] = double.NaN;
                    sessCompletedL[s] = double.NaN;
                    sessActive[s] = false;
                }

                // HTF swings (left=10, right=3)
                htfSwingHighs = new List<SwingPoint>();
                htfSwingLows = new List<SwingPoint>();

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
                bias = 0;
                overnightHigh = 0;
                overnightLow = double.MaxValue;
                ormHigh = 0;
                ormLow = double.MaxValue;
                ormLocked = false;
                ormBias = 0;

                lastNQSwingHighPrice = double.NaN;
                lastNQSwingLowPrice = double.NaN;
                lastESSwingHighPrice = double.NaN;
                lastESSwingLowPrice = double.NaN;
                currentSMTBull = false;
                currentSMTBear = false;

                newsEventTimes = new List<DateTime>();
                pendingSignal = null;
                pendingEarlyCut = false;
                nyOpenPrice = double.NaN;
                lastHTFBias = 0;
                htfPDACount = 0;
                rollDates = new HashSet<DateTime>();
                InitRollDates();
                LoadNewsCalendar();
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
                    Print(string.Format("=== LANTO FINAL === Trades={0} WR={1:F1}% TotalR={2:F1} MaxDD={3:F1}R PPDD={4:F1} CumR={5:F1}",
                        totalTrades, wr, totalR, maxDD, ppdd, cumR));
                    Print(string.Format("  Config: SkipLondon={0} SkipAsia={1} SMT={2} SQ={3} MinStopATR={4}",
                        SkipLondon, SkipAsia, SMTEnabled, SQEnabled, MinStopAtrMult));
                    Print(string.Format("  Bars: NY={0} London={1} Asia={2}",
                        dbgNYBars, dbgLondonBars, dbgAsiaBars));
                    Print(string.Format("  Signals: NY={0} London={1} Asia={2}",
                        dbgSignalsNY, dbgSignalsLondon, dbgSignalsAsia));
                    Print(string.Format("  Filtered: Session={0} Bias={1} SQ={2} MinStop={3}",
                        dbgFilteredSession, dbgFilteredBias, dbgFilteredSQ, dbgFilteredMinStop));
                    Print(string.Format("  SMT: BarsArrayLen={0} ES_swingH={1} ES_swingL={2} ES_bufSize={3}",
                        BarsArray.Length,
                        esSwingHighs != null ? esSwingHighs.Count : -1,
                        esSwingLows != null ? esSwingLows.Count : -1,
                        esHighBuffer != null ? esHighBuffer.Count : -1));
                }

                // === CSV TRADE LOG EXPORT ===
                // Auto-export trade log for Python comparison (Phase 5)
                if (tradeLog != null && tradeLog.Count > 0)
                {
                    string csvPath = System.IO.Path.Combine(
                        Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                        "nt8_trades_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".csv");
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

        #region Main Loop — OnBarUpdate

        protected override void OnBarUpdate()
        {
            // Guard: all series must have minimum bars before any processing
            if (CurrentBars[BIP_NQ5M] < 50) return;
            if (CurrentBars[BIP_NQ1H] < 3) return;
            if (CurrentBars[BIP_NQ4H] < 3) return;
            if (SMTEnabled && BarsArray.Length > BIP_ES5M && CurrentBars[BIP_ES5M] < 10) return;

            if (BarsInProgress == BIP_NQ5M)
                On5mBar();
            else if (BarsInProgress == BIP_NQ1H)
                On1hBar();
            else if (BarsInProgress == BIP_NQ4H)
                On4hBar();
            else if (BarsInProgress == BIP_ES5M)
                OnES5mBar();
        }

        private void On5mBar()
        {
            barIndex++;
            double av = atrIndicator[0];
            if (av <= 0 || double.IsNaN(av)) return;

            // Platform timezone = Eastern (set in NT8 Options > General > Time zone).
            // Time[0] = bar CLOSE time in ET. Subtract bar period to get OPEN time,
            // matching Python's parquet index exactly. No timezone conversion needed.
            DateTime barOpenET = Time[0].AddMinutes(-BarsPeriod.Value);
            DateTime barTimeET = barOpenET;  // all logic uses bar open time
            double hf = barOpenET.Hour + barOpenET.Minute / 60.0;
            string session = GetSession(hf);

            // Debug: count bars per session
            if (session == "ny") dbgNYBars++;
            else if (session == "london") dbgLondonBars++;
            else dbgAsiaBars++;

            // Daily reset
            NewDayCheck(barTimeET);

            // Track overnight range
            if (session == "asia" || session == "london")
            {
                if (High[0] > overnightHigh) overnightHigh = High[0];
                if (Low[0] < overnightLow) overnightLow = Low[0];
            }

            // ORM tracking (9:30-10:00 ET)
            if (!ormLocked && hf >= 9.5 && hf < 10.0)
            {
                if (High[0] > ormHigh) ormHigh = High[0];
                if (Low[0] < ormLow) ormLow = Low[0];
            }
            else if (!ormLocked && hf >= 10.0)
            {
                ormLocked = true;
                // ORM bias: broke above overnight high (only) = bull
                bool brokeAbove = ormHigh > overnightHigh && overnightHigh > 0;
                bool brokeBelow = ormLow < overnightLow && overnightLow < double.MaxValue;
                if (brokeAbove && !brokeBelow) ormBias = 1;
                else if (brokeBelow && !brokeAbove) ormBias = -1;
                else ormBias = 0;
            }

            // v8.1: Capture NY open price (first bar at 9:30 ET)
            if (double.IsNaN(nyOpenPrice) && hf >= 9.5)
                nyOpenPrice = Open[0];

            // Update session liquidity levels (Python _compute_liquidity_levels)
            UpdateSessionLiquidity(hf, High[0], Low[0]);

            // Update swings on NQ 5m (left=3, right=1)
            UpdateSwings(High[0], Low[0], barIndex, swingHighs, swingLows,
                         Highs[BIP_NQ5M], Lows[BIP_NQ5M], CurrentBars[BIP_NQ5M]);

            // Update HTF swings (left=10, right=3) for sweep scoring
            UpdateHTFSwings(High[0], Low[0], barIndex);

            // Step 1: Detect new FVGs ONLY (Fix #7: birth detection before signals)
            DetectNewFVG5m(av);

            // Compute SMT divergence
            if (SMTEnabled && BarsArray.Length > BIP_ES5M && CurrentBars[BIP_ES5M] > SMTSweepLookback)
                ComputeSMTDivergence(av);

            // Step 2: Handle pending early cut from previous bar (Fix #10)
            if (ts != null && pendingEarlyCut)
            {
                ExitTrade(Open[0], "early_cut_pa");
                pendingEarlyCut = false;
                // Don't return — allow new signal check below
            }

            // Step 3: Manage open position
            if (ts != null)
            {
                ManagePosition(av, hf, barTimeET);
                // After ManagePosition, if still in trade, skip signal check
                if (ts != null) goto PostSignals;
            }

            // Step 4: Execute pending signal from previous bar at this bar's open (Fix #3)
            if (pendingSignal != null && ts == null)
            {
                // Update entry price to current bar's open
                pendingSignal["entry_price"] = Open[0];
                double pendEntry = (double)pendingSignal["entry_price"];
                double pendStop = (double)pendingSignal["model_stop"];
                int pendDir = (int)pendingSignal["direction"];

                // v8.1: Recalculate TP1 with actual entry price (Fix: stale TP1 bug)
                pendingSignal["tp1"] = FindIRL(pendDir, pendEntry, pendStop);

                // Validate stop/entry with actual entry price
                bool stopValid = (pendDir == 1 && pendStop < pendEntry) ||
                                 (pendDir == -1 && pendStop > pendEntry);
                if (stopValid && PassesFilters(pendingSignal, av, session, hf, barTimeET))
                {
                    EnterTrade(pendingSignal, av, session, hf, barTimeET);
                    // v8.1: Mark FVG as used only after successful entry (Fix: wasted FVG bug)
                    if (pendingSignal.ContainsKey("_fvg_index"))
                    {
                        int fvgIdx = (int)pendingSignal["_fvg_index"];
                        if (fvgIdx >= 0 && fvgIdx < fvgs5m.Count)
                            fvgs5m[fvgIdx].Status = "used";
                    }
                }
                pendingSignal = null;
            }

            // If just entered a trade, skip signal detection
            if (ts != null) goto PostSignals;

            // ORM window: no entries 9:30-10:00 ET
            if (hf >= 9.5 && hf < 9.5 + ORMWindowMin / 60.0) goto PostSignals;

            // Lunch dead zone skip
            if (hf >= LunchStart && hf < LunchEnd) goto PostSignals;

            // Step 5: Check for new signals — saved as pending for next bar (Fix #3)
            if (!dayStopped)
            {
                var sig = CheckSignals(av, session, hf);
                if (sig != null)
                    pendingSignal = sig;
            }

        PostSignals:
            // Step 6: Update FVG states AFTER signal detection (Fix #7)
            UpdateFVGStates5m(av);
        }

        private void On1hBar()
        {
            if (CurrentBars[BIP_NQ1H] < 3) return;
            DetectHTFFVG(
                Highs[BIP_NQ1H][2], Lows[BIP_NQ1H][2],
                Opens[BIP_NQ1H][1], Highs[BIP_NQ1H][1], Lows[BIP_NQ1H][1], Closes[BIP_NQ1H][1],
                Highs[BIP_NQ1H][0], Lows[BIP_NQ1H][0],
                fvgs1h, 2000);

            // v8.1: Invalidate 1H FVGs that price has closed through
            double close1h = Closes[BIP_NQ1H][0];
            double high1h = Highs[BIP_NQ1H][0];
            double low1h = Lows[BIP_NQ1H][0];
            for (int f = fvgs1h.Count - 1; f >= 0; f--)
            {
                var fvg = fvgs1h[f];
                if (fvg.Status == "invalidated") continue;
                if (fvg.Direction == 1 && close1h < fvg.Bottom)
                    fvg.Status = "invalidated";
                else if (fvg.Direction == -1 && close1h > fvg.Top)
                    fvg.Status = "invalidated";
                // Also update untested -> tested_rejected
                if (fvg.Direction == 1 && low1h <= fvg.Top && fvg.Status == "untested")
                    fvg.Status = "tested_rejected";
                else if (fvg.Direction == -1 && high1h >= fvg.Bottom && fvg.Status == "untested")
                    fvg.Status = "tested_rejected";
            }
        }

        private void On4hBar()
        {
            if (CurrentBars[BIP_NQ4H] < 3) return;

            // Fix #4: Removed old 4H candle direction bias — now using full HTF FVG draw analysis
            // Fix: 4H FVGs go into fvgs4h (not fvgs1h)
            DetectHTFFVG(
                Highs[BIP_NQ4H][2], Lows[BIP_NQ4H][2],
                Opens[BIP_NQ4H][1], Highs[BIP_NQ4H][1], Lows[BIP_NQ4H][1], Closes[BIP_NQ4H][1],
                Highs[BIP_NQ4H][0], Lows[BIP_NQ4H][0],
                fvgs4h, 2000);

            // v8.1: Invalidate 4H FVGs that price has closed through
            double close4h = Closes[BIP_NQ4H][0];
            double high4h = Highs[BIP_NQ4H][0];
            double low4h = Lows[BIP_NQ4H][0];
            for (int f = fvgs4h.Count - 1; f >= 0; f--)
            {
                var fvg = fvgs4h[f];
                if (fvg.Status == "invalidated") continue;
                if (fvg.Direction == 1 && close4h < fvg.Bottom)
                    fvg.Status = "invalidated";
                else if (fvg.Direction == -1 && close4h > fvg.Top)
                    fvg.Status = "invalidated";
                // Also update untested -> tested_rejected
                if (fvg.Direction == 1 && low4h <= fvg.Top && fvg.Status == "untested")
                    fvg.Status = "tested_rejected";
                else if (fvg.Direction == -1 && high4h >= fvg.Bottom && fvg.Status == "untested")
                    fvg.Status = "tested_rejected";
            }
        }

        private void OnES5mBar()
        {
            if (CurrentBars[BIP_ES5M] < 10) return;

            // Track ES swings for SMT
            UpdateSwings(
                Highs[BIP_ES5M][0], Lows[BIP_ES5M][0], barIndex,
                esSwingHighs, esSwingLows,
                Highs[BIP_ES5M], Lows[BIP_ES5M], CurrentBars[BIP_ES5M]);

            // Buffer ES highs/lows for sweep detection
            esHighBuffer.Add(Highs[BIP_ES5M][0]);
            esLowBuffer.Add(Lows[BIP_ES5M][0]);
            if (esHighBuffer.Count > SMTSweepLookback + 10)
            {
                esHighBuffer.RemoveAt(0);
                esLowBuffer.RemoveAt(0);
            }

            // Update ES swing prices
            if (esSwingHighs.Count > 0)
                lastESSwingHighPrice = esSwingHighs[esSwingHighs.Count - 1].Price;
            if (esSwingLows.Count > 0)
                lastESSwingLowPrice = esSwingLows[esSwingLows.Count - 1].Price;
        }

        #endregion

        #region FVG Detection & State Machine

        /// <summary>
        /// Fix #7: Birth detection only — creates new FVG objects.
        /// Called BEFORE CheckSignals so new FVGs are in the list.
        /// Fix #1: Rollover filter — skip FVGs spanning contract roll dates.
        /// </summary>
        private void DetectNewFVG5m(double av)
        {
            if (CurrentBars[BIP_NQ5M] < 3) return;

            // Candle references: c1=[2], c2=[1], c3=[0] (NinjaTrader barsAgo)
            double c1High = High[2], c1Low = Low[2];
            double c2Open = Open[1], c2High = High[1], c2Low = Low[1], c2Close = Close[1];
            double c3High = High[0], c3Low = Low[0];

            // Fix #1: Rollover filter — skip FVGs that span contract roll dates
            DateTime barOpenET = Time[0].AddMinutes(-BarsPeriod.Value);
            DateTime c3Date = barOpenET.Date;
            DateTime c2Date = Time[1].AddMinutes(-BarsPeriod.Value).Date;
            DateTime c1Date = Time[2].AddMinutes(-BarsPeriod.Value).Date;
            if (rollDates.Contains(c1Date) || rollDates.Contains(c2Date) || rollDates.Contains(c3Date))
                return; // skip: FVG spans rollover

            // Bullish FVG: c1.High < c3.Low (gap above c1)
            if (c1High < c3Low)
            {
                double sz = c3Low - c1High;
                fvgs5m.Add(new FVG
                {
                    BarIndex = barIndex,
                    Direction = 1,
                    Top = c3Low,
                    Bottom = c1High,
                    Size = sz,
                    Candle2Open = c2Open,
                    CreationTime = Time[1],
                    SweepScore = ComputeSweepScore(c2High, c2Low, 1)
                });
            }

            // Bearish FVG: c1.Low > c3.High (gap below c1)
            if (c1Low > c3High)
            {
                double sz = c1Low - c3High;
                fvgs5m.Add(new FVG
                {
                    BarIndex = barIndex,
                    Direction = -1,
                    Top = c1Low,
                    Bottom = c3High,
                    Size = sz,
                    Candle2Open = c2Open,
                    CreationTime = Time[1],
                    SweepScore = ComputeSweepScore(c2High, c2Low, -1)
                });
            }
        }

        /// <summary>
        /// Fix #7: State update only — updates existing FVG statuses, spawns IFVGs, prunes.
        /// Called AFTER CheckSignals so signals see pre-update states.
        /// </summary>
        private void UpdateFVGStates5m(double av)
        {
            if (CurrentBars[BIP_NQ5M] < 3) return;

            double curHigh = High[0], curLow = Low[0], curClose = Close[0];
            for (int f = fvgs5m.Count - 1; f >= 0; f--)
            {
                var fvg = fvgs5m[f];
                if (fvg.Status == "invalidated" || fvg.Status == "used") continue;

                int effectiveDir = fvg.IsIFVG ? fvg.IFVGDirection : fvg.Direction;

                if (effectiveDir == 1) // Bullish (or Bull IFVG = support)
                {
                    // Test: price enters zone
                    if (curLow <= fvg.Top && fvg.Status == "untested")
                        fvg.Status = "tested_rejected";

                    // Invalidate: price closes below bottom
                    if (curClose < fvg.Bottom)
                    {
                        fvg.Status = "invalidated";
                        // Spawn IFVG (inversion: bull→bear)
                        if (!fvg.IsIFVG)
                        {
                            fvgs5m.Add(new FVG
                            {
                                BarIndex = barIndex,
                                Direction = 1,  // original direction
                                Top = fvg.Top,
                                Bottom = fvg.Bottom,
                                Size = fvg.Size,
                                Candle2Open = fvg.Top,  // zone top as stop for bear IFVG
                                CreationTime = fvg.CreationTime,
                                IsIFVG = true,
                                IFVGDirection = -1,  // inverted: now acts as resistance
                                SweepScore = fvg.SweepScore,
                                SweptLiquidity = fvg.SweptLiquidity
                            });
                        }
                    }
                }
                else // Bearish (or Bear IFVG = resistance)
                {
                    if (curHigh >= fvg.Bottom && fvg.Status == "untested")
                        fvg.Status = "tested_rejected";

                    if (curClose > fvg.Top)
                    {
                        fvg.Status = "invalidated";
                        if (!fvg.IsIFVG)
                        {
                            fvgs5m.Add(new FVG
                            {
                                BarIndex = barIndex,
                                Direction = -1,
                                Top = fvg.Top,
                                Bottom = fvg.Bottom,
                                Size = fvg.Size,
                                Candle2Open = fvg.Bottom,  // zone bottom as stop for bull IFVG
                                CreationTime = fvg.CreationTime,
                                IsIFVG = true,
                                IFVGDirection = 1,  // inverted: now acts as support
                                SweepScore = fvg.SweepScore,
                                SweptLiquidity = fvg.SweptLiquidity
                            });
                        }
                    }
                }
            }

            // Prune old FVGs (>500 bars)
            fvgs5m.RemoveAll(f => barIndex - f.BarIndex > 500);
        }

        private void DetectHTFFVG(double c1High, double c1Low,
            double c2Open, double c2High, double c2Low, double c2Close,
            double c3High, double c3Low,
            List<FVG> pool, int maxAge)
        {
            if (c1High < c3Low)
            {
                pool.Add(new FVG
                {
                    BarIndex = barIndex,
                    Direction = 1,
                    Top = c3Low,
                    Bottom = c1High,
                    Size = c3Low - c1High,
                    Candle2Open = c2Open,
                    CreationTime = Time[0]
                });
            }
            if (c1Low > c3High)
            {
                pool.Add(new FVG
                {
                    BarIndex = barIndex,
                    Direction = -1,
                    Top = c1Low,
                    Bottom = c3High,
                    Size = c1Low - c3High,
                    Candle2Open = c2Open,
                    CreationTime = Time[0]
                });
            }
            pool.RemoveAll(f => barIndex - f.BarIndex > maxAge);
        }

        private int ComputeSweepScore(double candleHigh, double candleLow, int dir)
        {
            // Python entry_signals.py lines 283-309: sweep scoring uses
            // 1) Completed session liquidity levels (+2 each)
            // 2) HTF swing levels (+2 each)
            // NOT dense 5m swings (too many false positives)
            int score = 0;

            // Session levels (completed sub-session H/L)
            for (int s = 0; s < 5; s++)
            {
                if (!double.IsNaN(sessCompletedH[s]) && candleHigh > sessCompletedH[s])
                    score += 2;
                if (!double.IsNaN(sessCompletedL[s]) && candleLow < sessCompletedL[s])
                    score += 2;
            }

            // HTF swing levels (left=10, right=3 — structural levels)
            if (htfSwingHighs.Count > 0)
            {
                double htfSH = htfSwingHighs[htfSwingHighs.Count - 1].Price;
                if (candleHigh > htfSH) score += 2;
            }
            if (htfSwingLows.Count > 0)
            {
                double htfSL = htfSwingLows[htfSwingLows.Count - 1].Price;
                if (candleLow < htfSL) score += 2;
            }

            return score;
        }

        private void UpdateSessionLiquidity(double hf, double barHigh, double barLow)
        {
            // Python _compute_liquidity_levels: 5 sub-sessions
            // 0=Asia(18-3 wraps), 1=London(3-9.5), 2=NY_AM(9.5-11), 3=NY_Lunch(11-13), 4=NY_PM(13-16)
            double[][] bounds = new double[][]
            {
                new double[] { 18.0, 3.0 },  // Asia (wraps midnight)
                new double[] { 3.0, 9.5 },
                new double[] { 9.5, 11.0 },
                new double[] { 11.0, 13.0 },
                new double[] { 13.0, 16.0 }
            };

            for (int s = 0; s < 5; s++)
            {
                bool nowIn;
                if (s == 0) // Asia wraps midnight
                    nowIn = hf >= bounds[s][0] || hf < bounds[s][1];
                else
                    nowIn = hf >= bounds[s][0] && hf < bounds[s][1];

                if (nowIn)
                {
                    if (!sessActive[s])
                    {
                        // Session just started — reset running
                        sessRunningH[s] = barHigh;
                        sessRunningL[s] = barLow;
                        sessActive[s] = true;
                    }
                    else
                    {
                        if (barHigh > sessRunningH[s]) sessRunningH[s] = barHigh;
                        if (barLow < sessRunningL[s]) sessRunningL[s] = barLow;
                    }
                }
                else
                {
                    if (sessActive[s])
                    {
                        // Session just ended — freeze as completed
                        sessCompletedH[s] = sessRunningH[s];
                        sessCompletedL[s] = sessRunningL[s];
                        sessActive[s] = false;
                    }
                }
            }
        }

        private void UpdateHTFSwings(double barHigh, double barLow, int bi)
        {
            // HTF swings: left=10, right=3 (Python entry_signals.py line 236)
            int htfLeft = 10;
            int htfRight = 3;
            int ci = bi - htfRight;
            if (ci < htfLeft || ci <= 0) return;

            // Need to access bars [ci-htfLeft .. ci+htfRight] via barsAgo
            // ci is the candidate bar. barsAgo = barIndex - ci.
            int ciBarsAgo = barIndex - ci;
            if (ciBarsAgo < 0 || ciBarsAgo >= CurrentBars[BIP_NQ5M]) return;

            double candidateH = High[ciBarsAgo];
            bool isSH = true;
            for (int j = ci - htfLeft; j < ci; j++)
            {
                int ba = barIndex - j;
                if (ba < 0 || ba >= CurrentBars[BIP_NQ5M]) { isSH = false; break; }
                if (High[ba] >= candidateH) { isSH = false; break; }
            }
            if (isSH)
            {
                for (int j = ci + 1; j <= ci + htfRight; j++)
                {
                    int ba = barIndex - j;
                    if (ba < 0 || ba >= CurrentBars[BIP_NQ5M]) { isSH = false; break; }
                    if (High[ba] >= candidateH) { isSH = false; break; }
                }
            }

            double candidateL = Low[ciBarsAgo];
            bool isSL = true;
            for (int j = ci - htfLeft; j < ci; j++)
            {
                int ba = barIndex - j;
                if (ba < 0 || ba >= CurrentBars[BIP_NQ5M]) { isSL = false; break; }
                if (Low[ba] <= candidateL) { isSL = false; break; }
            }
            if (isSL)
            {
                for (int j = ci + 1; j <= ci + htfRight; j++)
                {
                    int ba = barIndex - j;
                    if (ba < 0 || ba >= CurrentBars[BIP_NQ5M]) { isSL = false; break; }
                    if (Low[ba] <= candidateL) { isSL = false; break; }
                }
            }

            if (isSH)
            {
                htfSwingHighs.Add(new SwingPoint { BarIndex = ci, Price = candidateH });
                if (htfSwingHighs.Count > 50) htfSwingHighs.RemoveAt(0);
            }
            if (isSL)
            {
                htfSwingLows.Add(new SwingPoint { BarIndex = ci, Price = candidateL });
                if (htfSwingLows.Count > 50) htfSwingLows.RemoveAt(0);
            }
        }

        #endregion

        #region Signal Detection

        private Dictionary<string, object> CheckSignals(double av, string session, double hf)
        {
            if (CurrentBars[BIP_NQ5M] < 10) return null;

            // Check Trend signal first, then MSS
            var sig = CheckTrendSignal(av);
            if (sig != null) return sig;
            return CheckMSSSignal(av);
        }

        private Dictionary<string, object> CheckTrendSignal(double av)
        {
            // Match Python entry_signals.py exactly:
            // Single-bar rejection pattern (NOT 2-bar from QC port)
            double curClose = Close[0], curOpen = Open[0];
            double curHigh = High[0], curLow = Low[0];

            // Pre-check: rejection candle body ratio (skip entire check if candle weak)
            double bd = Math.Abs(curClose - curOpen);
            double rn = curHigh - curLow;
            if (rn <= 0 || bd / rn < FVGRejectionBodyRatio) return null;

            // Fluency pre-check
            // (Python checks fluency once, not per-FVG)
            // We'll check per direction below since fluency depends on direction

            // Best signal selection (Python picks highest scoring FVG)
            int bestDir = 0;
            FVG bestFVG = null;
            double bestScore = -1;

            foreach (var f in fvgs5m)
            {
                if (f.Status == "invalidated" || f.Status == "used") continue;
                if (f.IsIFVG) continue; // Trend only from regular FVGs
                if (f.BarIndex >= barIndex) continue; // FVG must be born before current bar
                if (barIndex - f.LastSignalIdx < FVGCooldown) continue;

                // FVG size filter
                if (av > 0 && f.Size < FVGMinSizeAtr * av) continue;

                int d = f.Direction;

                // FIX: Check displacement of CURRENT bar (Python checks disp_arr[i] at signal time)
                bool curDisplaced = (bd > 0 && rn > 0 && bd / rn >= DispBodyRatio && bd >= DispAtrMult * av);
                // Fix #6: Engulfment check — displacement candle must engulf at least 1 prior candle
                if (curDisplaced && CurrentBars[BIP_NQ5M] >= 2)
                {
                    bool engulfs = false;
                    double bodyHigh = Math.Max(curClose, curOpen);
                    double bodyLow = Math.Min(curClose, curOpen);
                    for (int k = 1; k <= Math.Min(3, CurrentBars[BIP_NQ5M] - 1); k++)
                    {
                        if (bodyHigh >= High[k] && bodyLow <= Low[k])
                        { engulfs = true; break; }
                    }
                    curDisplaced = engulfs;
                }

                if (d == 1) // Bull FVG
                {
                    // Single-bar: low enters zone AND close rejects above top
                    bool entered = curLow <= f.Top && curHigh >= f.Bottom;
                    bool rejected = curClose > f.Top;
                    if (entered && rejected)
                    {
                        // FIX: score = size + displacement(100) + sweep(200), matching Python line 384
                        double score = f.Size + (curDisplaced ? 100.0 : 0.0) + (f.SweptLiquidity ? 200.0 : 0.0);
                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestDir = 1;
                            bestFVG = f;
                        }
                    }
                }
                else // Bear FVG
                {
                    bool entered = curHigh >= f.Bottom && curLow <= f.Top;
                    bool rejected = curClose < f.Bottom;
                    if (entered && rejected)
                    {
                        double score = f.Size + (curDisplaced ? 100.0 : 0.0) + (f.SweptLiquidity ? 200.0 : 0.0);
                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestDir = -1;
                            bestFVG = f;
                        }
                    }
                }
            }

            if (bestFVG == null) return null;

            // Fluency check for winning direction
            if (!CheckFluency(bestDir, av)) return null;

            bestFVG.LastSignalIdx = barIndex;
            // Fix #3: Entry will be next bar's open — use Close[0] as placeholder
            double entry = curClose;
            double stop = bestFVG.Candle2Open;

            // Fix #3: Stop validation removed here — will be checked at pending execution
            // with actual entry price (next bar's open)

            // v8.1: Don't mark used yet — defer to execution (Fix: wasted FVG bug)
            // bestFVG.Status = "used";  // REMOVED
            double tp1 = FindIRL(bestDir, entry, stop);

            return new Dictionary<string, object>
            {
                {"direction", bestDir}, {"type", "trend"}, {"entry_price", entry},
                {"model_stop", stop}, {"tp1", tp1}, {"has_smt", false},
                {"sweep_score", bestFVG.SweepScore},
                {"_fvg_index", fvgs5m.IndexOf(bestFVG)}  // v8.1: store FVG index for deferred marking
            };
        }

        private Dictionary<string, object> CheckMSSSignal(double av)
        {
            // Match Python entry_signals.py MSS detection exactly
            double curClose = Close[0], curOpen = Open[0];
            double curHigh = High[0], curLow = Low[0];

            // Pre-check body ratio (same as Trend)
            double bd0 = Math.Abs(curClose - curOpen);
            double rn0 = curHigh - curLow;
            if (rn0 <= 0 || bd0 / rn0 < FVGRejectionBodyRatio) return null;

            int bestDir = 0;
            FVG bestIFVG = null;
            double bestScore = -1;

            foreach (var f in fvgs5m)
            {
                // MSS uses IFVGs only (invalidated FVGs that spawned inversions)
                if (!f.IsIFVG) continue;
                if (f.Status == "invalidated" || f.Status == "used") continue;
                if (f.BarIndex >= barIndex) continue; // must age 1 bar
                if (barIndex - f.LastSignalIdx < FVGCooldown) continue;
                if (av > 0 && f.Size < FVGMinSizeAtr * av) continue;

                int idir = f.IFVGDirection;

                // Python entry_signals.py line 614-618:
                // Check if swing level was ACTUALLY SWEPT (price went through it)
                // in the lookback window before IFVG birth.
                // swept_low[j] = low[j] < swing_low_price[j-1]
                bool hadSweep = false;
                int sweepStart = Math.Max(0, f.BarIndex - 20);
                if (idir == 1) // bull IFVG → need prior swing LOW sweep
                {
                    // Check if any bar in [sweepStart, f.BarIndex] had low < a swing low
                    foreach (var sl in swingLows)
                    {
                        if (sl.BarIndex >= sweepStart && sl.BarIndex < f.BarIndex)
                        {
                            // Check if any bar AFTER this swing was confirmed actually swept it
                            for (int bi = sl.BarIndex + 1; bi <= f.BarIndex && bi < CurrentBars[BIP_NQ5M]; bi++)
                            {
                                int barsAgo = barIndex - bi;
                                if (barsAgo >= 0 && barsAgo < CurrentBars[BIP_NQ5M] && Low[barsAgo] < sl.Price)
                                { hadSweep = true; break; }
                            }
                            if (hadSweep) break;
                        }
                    }
                }
                else // bear IFVG → need prior swing HIGH sweep
                {
                    foreach (var sh in swingHighs)
                    {
                        if (sh.BarIndex >= sweepStart && sh.BarIndex < f.BarIndex)
                        {
                            for (int bi = sh.BarIndex + 1; bi <= f.BarIndex && bi < CurrentBars[BIP_NQ5M]; bi++)
                            {
                                int barsAgo = barIndex - bi;
                                if (barsAgo >= 0 && barsAgo < CurrentBars[BIP_NQ5M] && High[barsAgo] > sh.Price)
                                { hadSweep = true; break; }
                            }
                            if (hadSweep) break;
                        }
                    }
                }
                if (!hadSweep) continue;

                // Single-bar retest + rejection (matching Python exactly)
                if (idir == 1) // Bull IFVG: low enters, close above top
                {
                    bool entered = curLow <= f.Top && curHigh >= f.Bottom;
                    bool rejected = curClose > f.Top;
                    if (entered && rejected)
                    {
                        double score = f.Size;
                        if (score > bestScore) { bestScore = score; bestDir = 1; bestIFVG = f; }
                    }
                }
                else // Bear IFVG: high enters, close below bottom
                {
                    bool entered = curHigh >= f.Bottom && curLow <= f.Top;
                    bool rejected = curClose < f.Bottom;
                    if (entered && rejected)
                    {
                        double score = f.Size;
                        if (score > bestScore) { bestScore = score; bestDir = -1; bestIFVG = f; }
                    }
                }
            }

            if (bestIFVG == null) return null;
            if (!CheckFluency(bestDir, av)) return null;

            double entry = curClose;
            double stop = bestDir == 1 ? bestIFVG.Bottom : bestIFVG.Top;
            // Fix #3: Stop validation removed — checked at pending execution with actual entry

            // v8.1: Don't mark used yet — defer to execution (Fix: wasted FVG bug)
            // bestIFVG.Status = "used";  // REMOVED
            double tp1 = FindIRL(bestDir, entry, stop);

            // SMT gating
            bool hasSMT = false;
            if (SMTEnabled)
                hasSMT = (bestDir == 1 && currentSMTBull) || (bestDir == -1 && currentSMTBear);
            if (SMTRequireForMSS && !hasSMT) return null;

            return new Dictionary<string, object>
            {
                {"direction", bestDir}, {"type", "mss"}, {"entry_price", entry},
                {"model_stop", stop}, {"tp1", tp1}, {"has_smt", hasSMT},
                {"sweep_score", bestIFVG.SweepScore},
                {"_fvg_index", fvgs5m.IndexOf(bestIFVG)}  // v8.1: store FVG index for deferred marking
            };
        }

        /// <summary>
        /// Fix #2: Fluency uses max(bull,bear)/window (direction-agnostic majority count),
        /// matching Python implementation exactly.
        /// </summary>
        private bool CheckFluency(int dir, double av)
        {
            int n = Math.Min(FluencyWindow, CurrentBars[BIP_NQ5M]);
            if (n < 3) return false;

            int bullCount = 0, bearCount = 0;
            double sumBodyRatio = 0;
            double sumBarSizeVsATR = 0;
            int validBars = 0;

            for (int j = 0; j < n; j++)
            {
                double body = Math.Abs(Close[j] - Open[j]);
                double range = High[j] - Low[j];
                if (range <= 0) continue;

                if (Close[j] > Open[j]) bullCount++;
                else bearCount++;

                sumBodyRatio += body / range;
                sumBarSizeVsATR += Math.Min(1.0, range / (av > 0 ? av : 1));
                validBars++;
            }

            if (validBars < 3) return false;

            double dirRatio = (double)Math.Max(bullCount, bearCount) / n;
            double avgBodyRatio = sumBodyRatio / validBars;
            double avgBarSize = sumBarSizeVsATR / validBars;

            double fluency = FluencyWDir * dirRatio + FluencyWBody * avgBodyRatio + FluencyWBar * avgBarSize;
            return fluency >= FluencyThreshold;
        }

        private double ComputeSignalQuality(Dictionary<string, object> sig, double av)
        {
            double entry = (double)sig["entry_price"];
            double stop = (double)sig["model_stop"];

            // 1. Size: gap distance / (ATR * 1.5)
            double gap = Math.Abs(entry - stop);
            double sizeSc = av > 0 ? Math.Min(1.0, gap / (av * 1.5)) : 0.5;

            // 2. Displacement: current candle body/range
            double body = Math.Abs(Close[0] - Open[0]);
            double range = High[0] - Low[0];
            double dispSc = range > 0 ? body / range : 0;

            // 3. Fluency (recompute as score)
            double fluSc = ComputeFluencyScore((int)sig["direction"], av);

            // 4. PA cleanliness: 1 - alternating direction ratio
            double paSc = ComputePACleanliness();

            return SQWSize * sizeSc + SQWDisp * dispSc + SQWFlu * fluSc + SQWPA * paSc;
        }

        /// <summary>
        /// Fix #2 + #11: ComputeFluencyScore uses max(bull,bear)/n and actual avgBarSize
        /// instead of hardcoded 0.5, matching Python.
        /// </summary>
        private double ComputeFluencyScore(int dir, double av)
        {
            int n = Math.Min(FluencyWindow, CurrentBars[BIP_NQ5M]);
            if (n < 3) return 0.5;

            int bullCount = 0, bearCount = 0;
            double sumBodyRatio = 0;
            double sumBarSizeVsATR = 0;
            int validBars = 0;

            for (int j = 0; j < n; j++)
            {
                double body = Math.Abs(Close[j] - Open[j]);
                double range = High[j] - Low[j];
                if (range <= 0) continue;

                if (Close[j] > Open[j]) bullCount++;
                else bearCount++;

                sumBodyRatio += body / range;
                sumBarSizeVsATR += Math.Min(1.0, range / (av > 0 ? av : 1));
                validBars++;
            }
            if (validBars < 3) return 0.5;

            double fluency = FluencyWDir * ((double)Math.Max(bullCount, bearCount) / n) +
                             FluencyWBody * (sumBodyRatio / validBars) +
                             FluencyWBar * (sumBarSizeVsATR / validBars);
            return Math.Min(1.0, Math.Max(0.0, fluency));
        }

        private double ComputePACleanliness()
        {
            int window = 6;
            int n = Math.Min(window, CurrentBars[BIP_NQ5M] - 1);
            if (n < 3) return 0.5;

            int altCount = 0;
            for (int j = 0; j < n - 1; j++)
            {
                int dir1 = Close[j + 1] > Open[j + 1] ? 1 : -1;
                int dir0 = Close[j] > Open[j] ? 1 : -1;
                if (dir1 != dir0) altCount++;
            }
            return 1.0 - (double)altCount / (n - 1);
        }

        /// <summary>
        /// Fix #8: Dynamic fallback uses risk * 2.0 instead of fixed +20.
        /// </summary>
        private double FindIRL(int dir, double entry, double stop)
        {
            double risk = Math.Abs(entry - stop);
            if (dir == 1)
            {
                // Nearest swing high above entry
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

        #endregion

        #region SMT Divergence

        private void ComputeSMTDivergence(double av)
        {
            // Update NQ swing prices
            if (swingHighs.Count > 0)
                lastNQSwingHighPrice = swingHighs[swingHighs.Count - 1].Price;
            if (swingLows.Count > 0)
                lastNQSwingLowPrice = swingLows[swingLows.Count - 1].Price;

            if (double.IsNaN(lastNQSwingHighPrice) || double.IsNaN(lastNQSwingLowPrice) ||
                double.IsNaN(lastESSwingHighPrice) || double.IsNaN(lastESSwingLowPrice))
            {
                currentSMTBull = false;
                currentSMTBear = false;
                return;
            }

            // NQ sweep detection: rolling max/min over lookback
            bool nqSweptHigh = false, nqSweptLow = false;
            int lb = Math.Min(SMTSweepLookback, CurrentBars[BIP_NQ5M]);
            for (int j = 0; j < lb; j++)
            {
                if (High[j] > lastNQSwingHighPrice) nqSweptHigh = true;
                if (Low[j] < lastNQSwingLowPrice) nqSweptLow = true;
            }

            // ES sweep detection from buffer
            bool esSweptHigh = false, esSweptLow = false;
            int esLB = Math.Min(SMTSweepLookback + SMTTimeTolerance, esHighBuffer.Count);
            for (int j = esHighBuffer.Count - esLB; j < esHighBuffer.Count; j++)
            {
                if (j < 0) continue;
                if (esHighBuffer[j] > lastESSwingHighPrice) esSweptHigh = true;
                if (esLowBuffer[j] < lastESSwingLowPrice) esSweptLow = true;
            }

            // SMT divergence: NQ swept but ES didn't
            currentSMTBull = nqSweptLow && !esSweptLow;
            currentSMTBear = nqSweptHigh && !esSweptHigh;
        }

        #endregion

        #region Bias Computation

        /// <summary>
        /// Fix #4: Full HTF FVG draw analysis for bias, replacing simple 4H candle direction.
        /// Counts active (non-invalidated) FVGs above/below price on 4H and 1H.
        /// </summary>
        private int ComputeHTFBias()
        {
            double curClose = Close[0];

            // Count active FVGs above/below price on 4H
            int bull4hAbove = 0, bear4hBelow = 0;
            foreach (var f in fvgs4h)
            {
                if (f.Status == "invalidated") continue;
                double mid = (f.Top + f.Bottom) / 2.0;
                if (f.Direction == 1 && mid > curClose) bull4hAbove++;
                else if (f.Direction == -1 && mid < curClose) bear4hBelow++;
            }

            // Count active FVGs above/below price on 1H
            int bull1hAbove = 0, bear1hBelow = 0;
            foreach (var f in fvgs1h)
            {
                if (f.Status == "invalidated") continue;
                double mid = (f.Top + f.Bottom) / 2.0;
                if (f.Direction == 1 && mid > curClose) bull1hAbove++;
                else if (f.Direction == -1 && mid < curClose) bear1hBelow++;
            }

            // Weighted combination (Python: w_4h=0.6, w_1h=0.4)
            double draw4h = 0;
            if (bull4hAbove > 0 && bear4hBelow == 0) draw4h = 1.0;
            else if (bear4hBelow > 0 && bull4hAbove == 0) draw4h = -1.0;
            else if (bull4hAbove > 0 && bear4hBelow > 0)
            {
                int net = bull4hAbove - bear4hBelow;
                draw4h = net > 0 ? 0.5 : (net < 0 ? -0.5 : 0.0);
            }

            double draw1h = 0;
            if (bull1hAbove > 0 && bear1hBelow == 0) draw1h = 1.0;
            else if (bear1hBelow > 0 && bull1hAbove == 0) draw1h = -1.0;
            else if (bull1hAbove > 0 && bear1hBelow > 0)
            {
                int net = bull1hAbove - bear1hBelow;
                draw1h = net > 0 ? 0.5 : (net < 0 ? -0.5 : 0.0);
            }

            double htfBias = 0.6 * draw4h + 0.4 * draw1h;
            lastHTFBias = htfBias;
            htfPDACount = bull4hAbove + bear4hBelow + bull1hAbove + bear1hBelow;
            return 0; // used in composite via lastHTFBias
        }

        private int GetCompositeBias()
        {
            // Compute HTF bias from FVG draw analysis
            ComputeHTFBias();

            // Overnight bias: position within overnight range (v8.1: use NY open, not live Close[0])
            int overnightBias = 0;
            if (double.IsNaN(nyOpenPrice))
            {
                overnightBias = 0;  // v8.1: no NY open yet, neutral
            }
            else if (overnightHigh > 0 && overnightLow < double.MaxValue)
            {
                double range = overnightHigh - overnightLow;
                if (range > 0)
                {
                    double pos = (nyOpenPrice - overnightLow) / range;  // v8.1: was Close[0]
                    if (pos > 0.6) overnightBias = 1;
                    else if (pos < 0.4) overnightBias = -1;
                }
            }

            double composite = 0.4 * lastHTFBias + 0.3 * overnightBias + 0.3 * ormBias;
            if (Math.Abs(composite) > 0.2)
                return composite > 0 ? 1 : -1;
            return 0;
        }

        #endregion

        #region Filters

        private bool PassesFilters(Dictionary<string, object> sig, double av,
            string session, double hf, DateTime barTimeET)
        {
            int dir = (int)sig["direction"];
            double entry = (double)sig["entry_price"];
            double stop = (double)sig["model_stop"];
            string sigType = (string)sig["type"];
            bool hasSMT = (bool)sig["has_smt"];
            bool isMSSSMT = sigType == "mss" && hasSMT && SMTEnabled;

            // Debug: count signals per session
            if (session == "ny") dbgSignalsNY++;
            else if (session == "london") dbgSignalsLondon++;
            else dbgSignalsAsia++;

            // FILTER 1: Bias opposing
            int biasCur = GetCompositeBias();
            if (biasCur != 0 && dir == -Math.Sign(biasCur))
            {
                if (isMSSSMT) { /* pass */ }
                else { dbgFilteredBias++; return false; }
            }

            // FILTER 1c: Session filter
            bool mssSessionBypass = isMSSSMT && SMTBypassSession;
            if (!mssSessionBypass)
            {
                if (session == "asia") { dbgFilteredSession++; return false; }
                if (session == "london" && SkipLondon) { dbgFilteredSession++; return false; }
            }

            // FILTER 2: Min stop distance (ATR-relative)
            double stopDist = Math.Abs(entry - stop);
            if (stopDist < MinStopAtrMult * av) { dbgFilteredMinStop++; return false; }

            // FILTER 3: Signal quality score
            if (SQEnabled)
            {
                double sq = ComputeSignalQuality(sig, av);
                double threshold = dir == 1 ? SQLongThreshold : SQShortThreshold;
                if (sq < threshold) { dbgFilteredSQ++; return false; }
            }

            // FILTER 4: News blackout
            if (IsInNewsBlackout(barTimeET)) return false;

            // FILTER 5: Daily limits
            if (consecutiveLosses >= MaxConsecLosses) return false;
            if (dailyPnlR <= -DailyMaxLossR) return false;

            // FILTER 6: Already in position
            if (ts != null) return false;

            return true;
        }

        private bool IsInNewsBlackout(DateTime barTimeET)
        {
            if (newsEventTimes == null || newsEventTimes.Count == 0) return false;

            foreach (var evt in newsEventTimes)
            {
                double minBefore = (evt - barTimeET).TotalMinutes;
                double minAfter = (barTimeET - evt).TotalMinutes;

                // Within 60 min before or 5 min after
                if (minBefore >= 0 && minBefore <= 60) return true;
                if (minAfter >= 0 && minAfter <= 5) return true;
            }
            return false;
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

            // Grade determination (simplified: use bias alignment + regime proxy)
            string grade = ComputeGrade(dir);

            // Position sizing
            bool isReduced = barTimeET.DayOfWeek == DayOfWeek.Monday ||
                             barTimeET.DayOfWeek == DayOfWeek.Friday;
            double baseR = isReduced ? ReducedR : NormalR;
            double rAmount;
            if (grade == "A+") rAmount = baseR * APlusMult;
            else if (grade == "B+") rAmount = baseR * BPlusMult;
            else rAmount = baseR * 0.5;

            // DD-based sizing removed — Python does NOT have this
            if (rAmount <= 0) return;

            int contracts = Math.Max(1, (int)(rAmount / (stopDist * PointValue)));

            // TP adjustment
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

            // Determine trim percentage
            double trimPctLocal;
            if (dir == -1) // all shorts: 100% trim (scalp)
                trimPctLocal = 1.0;
            else if (isMSS)
                trimPctLocal = TrimPct; // MSS long: default 50%
            else
                trimPctLocal = TrimPct; // Trend long: default 50%

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

            // No EOD close — Python engine keeps positions open overnight
            // (matches engine.py behavior exactly)
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
                // Full exit at TP1
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
                // Partial: trim already happened at TP1
                double remainRatio = (double)ts.Contracts / ts.OrigContracts;
                double remainR = ts.OrigStopDist > 0 ? (pp / ts.OrigStopDist) * remainRatio : 0;
                rm = ts.TrimR + remainR;
                if (reason == "stop" && Math.Abs(pp) < 1) reason = "be_sweep";
            }
            else if (reason == "tp1")
            {
                // Full exit at TP1
                double tpDist = Math.Abs(ts.TP1Price - ts.EntryPrice);
                rm = ts.OrigStopDist > 0 ? tpDist / ts.OrigStopDist : 0;
            }
            else
            {
                rm = ts.OrigStopDist > 0 ? pp / ts.OrigStopDist : 0;
            }

            // FIX: R-multiple should NOT be multiplied by SizeMult.
            // Python: R = total_pnl / total_risk (both scale with contracts, ratio is pure).
            // DD sizing affects contracts (position size) but NOT R-multiple.
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
                    {"r_multiple", rm},  // FIX: use raw R, not SizeMult-adjusted
                    {"exit_reason", reason},
                    {"grade", ts.Grade},
                    {"trimmed", ts.Trimmed}
                });
            }

            ts = null;
        }

        #endregion

        #region Position Sizing

        /// <summary>
        /// Fix #5: Regime computation uses htfPDACount from ComputeHTFBias().
        /// </summary>
        private string ComputeGrade(int dir)
        {
            int biasCur = GetCompositeBias();
            bool aligned = (dir == Math.Sign(biasCur) && biasCur != 0);

            // Regime: based on PDA count (from ComputeHTFBias)
            // 1.0 if has PDAs + clear bias, 0.5 if has PDAs only, 0.0 if no PDAs
            double regime;
            if (htfPDACount == 0) regime = 0.0;
            else regime = (biasCur != 0) ? 1.0 : 0.5;

            if (aligned && regime >= 1.0) return "A+";
            if (aligned || regime >= 1.0) return "B+";
            return "C";
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
                overnightHigh = 0;
                overnightLow = double.MaxValue;
                ormHigh = 0;
                ormLow = double.MaxValue;
                pendingSignal = null;
                pendingEarlyCut = false;
                ormLocked = false;
                ormBias = 0;
                nyOpenPrice = double.NaN;  // v8.1: reset NY open each day
            }
        }

        private void UpdateSwings(double high, double low, int idx,
            List<SwingPoint> shList, List<SwingPoint> slList,
            ISeries<double> highSeries, ISeries<double> lowSeries, int currentBar)
        {
            // Fractal swing detection: left=SwingLeftBars, right=SwingRightBars
            // Check bar at [SwingRightBars] (confirmed after right bars close)
            int right = SwingRightBars;
            int left = SwingLeftBars;
            if (currentBar < left + right + 1) return;

            double candidateHigh = highSeries[right];
            double candidateLow = lowSeries[right];

            // Check swing high: candidate > all left bars AND > all right bars
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
        /// Fix #9: Simply find nth most recent swing, no filtering by entry bar or price.
        /// Matches Python behavior.
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
        /// Fix #1: Initialize rollover dates for NQ quarterly rolls (2016-2026).
        /// NQ rolls on the 2nd Thursday of March, June, Sept, Dec.
        /// Also adds day before and after for safety.
        /// </summary>
        private void InitRollDates()
        {
            int[] rollMonths = { 3, 6, 9, 12 };
            for (int year = 2016; year <= 2026; year++)
            {
                foreach (int month in rollMonths)
                {
                    // Find 2nd Thursday: first day of month, find first Thursday, add 7 days
                    var firstDay = new DateTime(year, month, 1);
                    int daysToThursday = ((int)DayOfWeek.Thursday - (int)firstDay.DayOfWeek + 7) % 7;
                    var secondThursday = firstDay.AddDays(daysToThursday + 7);
                    rollDates.Add(secondThursday.Date);
                    rollDates.Add(secondThursday.AddDays(-1).Date);
                    rollDates.Add(secondThursday.AddDays(1).Date);
                }
            }
        }

        private void LoadNewsCalendar()
        {
            if (string.IsNullOrEmpty(NewsCSVPath) || !File.Exists(NewsCSVPath))
                return;

            try
            {
                var lines = File.ReadAllLines(NewsCSVPath);
                foreach (var line in lines.Skip(1)) // skip header
                {
                    var parts = line.Split(',');
                    if (parts.Length < 4) continue;

                    string impact = parts[3].Trim().ToLower();
                    if (impact != "high") continue;

                    string dateStr = parts[0].Trim();
                    string timeStr = parts[1].Trim();

                    if (DateTime.TryParse(dateStr + " " + timeStr, out DateTime eventTime))
                    {
                        // Assume time is ET
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
