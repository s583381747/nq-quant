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
    /// Lanto NQ Quantitative Strategy — v9 Standalone.
    /// VERSION: v9.1 (2026-04-02) — Signal detection fix patch
    ///
    /// Complete standalone NinjaTrader 8 strategy with both CSV-mode and internal signal
    /// detection. Trade management from LantoNQStrategy_CSV.cs (validated).
    /// Signal detection translated method-by-method from bar_by_bar_engine.py (2881 lines).
    ///
    /// v9.1 FIXES (signal detection divergence from Python baseline):
    ///   FIX 1: Removed FVG "used" status — FVGs now survive after first signal,
    ///          gated only by cooldown (last_signal_idx). Python never permanently
    ///          kills FVGs. This was the #1 source of missing ~9000 raw signals.
    ///   FIX 2: ORM skip window includes 10:00 bar (hf &lt;= 10.0 not &lt; 10.0),
    ///          matching Python's skip of (et_h==10 and et_m==0).
    ///   FIX 3: PA score in SQ computation now EXCLUDES current bar from window,
    ///          matching Python export's c[idx-6:idx] slice.
    ///   FIX 4: Displacement engulfment checks only 1 prior bar (engulf_min=1),
    ///          matching Python's engulf_min_candles=1 from params.yaml.
    ///   FIX 5: ComputeGrade regime uses abs(htf_bias) &gt; 0.2 (HTF only),
    ///          not composite bias. Matches Python _get_composite_bias().
    ///   FIX 6: htfPDACount uses total active HTF FVG count (all directions),
    ///          not just above/below price. Matches Python len(fvgs).
    ///
    /// Architecture:
    ///   UseCSVSignals=true  -> reads signals from CSV (same as LantoNQStrategy_CSV)
    ///   UseCSVSignals=false -> detects signals internally (translated from Python)
    ///
    /// Data series:
    ///   BIP 0: NQ 5min (primary)
    ///   BIP 1: NQ 1H (HTF bias)
    ///   BIP 2: NQ 4H (HTF bias)
    ///   BIP 3: ES 5min (SMT divergence)
    /// </summary>
    public class LantoNQStrategy_v9 : Strategy
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
            public string Status;       // untested, tested_rejected, invalidated, used
            public DateTime CreationTime;
            public int LastSignalIdx;
            public bool IsIFVG;
            public int IFVGDirection;   // direction after inversion
            public bool SweptLiquidity;
            public int SweepScore;
            public int InvalidatedAtIdx;

            public FVG()
            {
                Status = "untested";
                LastSignalIdx = -999;
                InvalidatedAtIdx = -1;
            }
        }

        private class HTF_FVG
        {
            public int BarIndex;
            public int Direction;       // +1 bull, -1 bear
            public double Top;
            public double Bottom;
            public double Size;
            public string Status;

            public HTF_FVG()
            {
                Status = "untested";
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

        private struct BarData
        {
            public double Open, High, Low, Close;
            public bool IsRollDate;
            public int BarIdx;
        }

        private class SignalRow
        {
            public int Direction;
            public string Type;
            public double EntryPrice;
            public double ModelStop;
            public double IRL_Target;
            public bool HasSMT;
            public string Grade;
            public int BiasDirection;
            public double Regime;
            public int SweepScore;
        }

        #endregion

        #region Parameters — Mode

        [NinjaScriptProperty]
        [Display(Name = "Use CSV Signals", Order = 0, GroupName = "0. Mode")]
        public bool UseCSVSignals { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Signal CSV Path", Order = 1, GroupName = "0. Mode")]
        public string SignalCSVPath { get; set; }

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

        [NinjaScriptProperty]
        [Display(Name = "Sweep Lookback", Order = 4, GroupName = "4. FVG")]
        public int SweepLookback { get; set; }

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
        [Display(Name = "Lunch Start (ET frac)", Order = 3, GroupName = "7. Sessions")]
        public double LunchStart { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Lunch End (ET frac)", Order = 4, GroupName = "7. Sessions")]
        public double LunchEnd { get; set; }

        #endregion

        #region Parameters — Signal Quality

        [NinjaScriptProperty]
        [Display(Name = "SQ Enabled", Order = 1, GroupName = "8. Signal Quality")]
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

        #region Parameters — Dual Mode / MSS

        [NinjaScriptProperty]
        [Display(Name = "Short RR (Trend)", Order = 1, GroupName = "9. Dual Mode")]
        public double DualShortRR { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "NY TP Multiplier", Order = 2, GroupName = "9. Dual Mode")]
        public double NYTPMult { get; set; }

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

        [NinjaScriptProperty]
        [Display(Name = "ES Instrument", Order = 6, GroupName = "11. SMT")]
        public string ESInstrument { get; set; }

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
        [Display(Name = "Min Stop ATR Mult", Order = 1, GroupName = "13. Misc")]
        public double MinStopAtrMult { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "A+ Size Mult", Order = 2, GroupName = "13. Misc")]
        public double APlusMult { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "B+ Size Mult", Order = 3, GroupName = "13. Misc")]
        public double BPlusMult { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Slippage Ticks", Order = 4, GroupName = "13. Misc")]
        public int SlippageTicks { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Commission/Side ($)", Order = 5, GroupName = "13. Misc")]
        public double CommissionPerSide { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "News CSV Path", Order = 6, GroupName = "13. Misc")]
        public string NewsCSVPath { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "PA Alt Dir Threshold", Order = 7, GroupName = "13. Misc")]
        public double PAThreshold { get; set; }

        #endregion

        #region Parameters — Early Cut

        [NinjaScriptProperty]
        [Display(Name = "Early Cut Min Bars", Order = 1, GroupName = "14. Early Cut")]
        public int EarlyCutMinBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Early Cut Max Bars", Order = 2, GroupName = "14. Early Cut")]
        public int EarlyCutMaxBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Early Cut Wick Ratio", Order = 3, GroupName = "14. Early Cut")]
        public double EarlyCutWickRatio { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Early Cut PA Threshold", Order = 4, GroupName = "14. Early Cut")]
        public double EarlyCutPAThreshold { get; set; }

        #endregion

        #region State Variables

        // Data series indices (BarsInProgress)
        private const int BIP_NQ5M = 0;
        private const int BIP_NQ1H = 1;
        private const int BIP_NQ4H = 2;
        private const int BIP_ES5M = 3;

        // Indicators
        private ATR atrIndicator;

        // Candle ring buffer (maxlen=20, matches Python deque)
        private List<BarData> candleBuffer;
        private const int CANDLE_BUFFER_MAX = 20;

        // FVG pools (5m)
        private List<FVG> fvgs5m;

        // HTF FVG pools (1H, 4H) for bias
        private List<HTF_FVG> htfFVGs1h;
        private List<HTF_FVG> htfFVGs4h;

        // Swing points (NQ 5m)
        private List<SwingPoint> swingHighs;
        private List<SwingPoint> swingLows;

        // HTF swings (left=10, right=3 on 5m data, for sweep scoring)
        private List<SwingPoint> htfSwingHighs;
        private List<SwingPoint> htfSwingLows;

        // ES swings for SMT
        private List<SwingPoint> esSwingHighs;
        private List<SwingPoint> esSwingLows;
        private List<double> esHighBuffer;
        private List<double> esLowBuffer;

        // SMT state
        private double lastNQSwingHighPrice;
        private double lastNQSwingLowPrice;
        private double lastESSwingHighPrice;
        private double lastESSwingLowPrice;
        private bool currentSMTBull;
        private bool currentSMTBear;

        // Sweep tracking buffers (for MSS: idx, swept)
        private List<KeyValuePair<int, bool>> sweptLowBuffer;
        private List<KeyValuePair<int, bool>> sweptHighBuffer;
        private const int SWEPT_BUFFER_MAX = 100;

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

        // HTF Bias state
        private double htfBias4h;
        private double htfBias1h;
        private int htfPDACount;

        // Session liquidity tracking (5 sub-sessions)
        // 0=Asia(18-3 wraps), 1=London(3-9.5), 2=NY_AM(9.5-11), 3=NY_Lunch(11-13), 4=NY_PM(13-16)
        private double[] sessRunningH;
        private double[] sessRunningL;
        private double[] sessCompletedH;
        private double[] sessCompletedL;
        private bool[] sessActive;

        // Overnight state
        private double overnightHigh;
        private double overnightLow;
        private double overnightRunningH;
        private double overnightRunningL;
        private bool inOvernight;

        // ORM state
        private double ormHigh;
        private double ormLow;
        private double ormRunningH;
        private double ormRunningL;
        private bool inORM;
        private double ormBias;
        private bool ormBiasLocked;
        private double nyOpenPrice;
        private bool nyOpenLocked;

        // News blackout
        private List<DateTime> newsEventTimes;

        // Pending signal mechanism (signal bar N -> execute bar N+1)
        private Dictionary<string, object> pendingSignal;

        // Pending early cut
        private bool pendingEarlyCut;

        // Rollover dates
        private HashSet<DateTime> rollDates;

        // Trade log for CSV export
        private List<Dictionary<string, object>> tradeLog;

        // CSV mode: signal map
        private Dictionary<long, SignalRow> signalMap;

        // Debug counters
        private int dbgSignalsFound;
        private int dbgSignalsExecuted;
        private int dbgFilteredSession, dbgFilteredBias, dbgFilteredSQ, dbgFilteredMinStop;
        private int dbgNYBars, dbgLondonBars, dbgAsiaBars;
        private int dbgTrendSignals, dbgMSSSignals;

        #endregion

        #region Initialization

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Lanto NQ Strategy v9.1 — Standalone (CSV + Internal Signals, 6 signal fixes)";
                Name = "LantoNQStrategy_v9";
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

                // Mode defaults
                UseCSVSignals = false;
                SignalCSVPath = "";

                // Displacement (from params.yaml)
                DispAtrMult = 0.8;
                DispBodyRatio = 0.60;

                // Fluency
                FluencyWindow = 6;
                FluencyThreshold = 0.60;
                FluencyWDir = 0.4;
                FluencyWBody = 0.3;
                FluencyWBar = 0.3;

                // Swing
                SwingLeftBars = 3;
                SwingRightBars = 1;

                // FVG
                FVGMinSizeAtr = 0.3;
                FVGCooldown = 6;
                FVGRejectionBodyRatio = 0.50;
                SweepLookback = 20;

                // Position sizing
                NormalR = 1000;
                ReducedR = 500;
                PointValue = 2;  // MNQ micro

                // Risk
                DailyMaxLossR = 2.0;
                MaxConsecLosses = 2;

                // Sessions
                SkipLondon = true;
                SkipAsia = true;
                LunchStart = 12.5;
                LunchEnd = 13.0;

                // Signal quality
                SQEnabled = true;
                SQLongThreshold = 0.68;
                SQShortThreshold = 0.82;
                SQWSize = 0.30;
                SQWDisp = 0.30;
                SQWFlu = 0.20;
                SQWPA = 0.20;

                // Dual mode
                DualShortRR = 0.625;
                NYTPMult = 2.0;

                // MSS
                MSSLongTPMult = 2.5;
                MSSShortRR = 0.50;

                // SMT
                SMTEnabled = true;
                SMTSweepLookback = 15;
                SMTTimeTolerance = 1;
                SMTRequireForMSS = true;
                SMTBypassSession = true;
                ESInstrument = "ES_5M_NT8";

                // Trail
                TrimPct = 0.50;
                NthSwingTrail = 2;

                // Misc
                MinStopAtrMult = 1.7;
                APlusMult = 1.5;
                BPlusMult = 1.0;
                SlippageTicks = 1;
                CommissionPerSide = 0.62;
                NewsCSVPath = "";
                PAThreshold = 1.0;

                // Early cut
                EarlyCutMinBars = 3;
                EarlyCutMaxBars = 4;
                EarlyCutWickRatio = 0.65;
                EarlyCutPAThreshold = 0.3;
            }
            else if (State == State.Configure)
            {
                // Always add HTF series for bias computation (even in CSV mode)
                AddDataSeries(BarsPeriodType.Minute, 60);    // BIP 1: NQ 1H
                AddDataSeries(BarsPeriodType.Minute, 240);   // BIP 2: NQ 4H

                // ES 5m for SMT divergence
                if (SMTEnabled && !string.IsNullOrEmpty(ESInstrument))
                    AddDataSeries(ESInstrument, BarsPeriodType.Minute, 5); // BIP 3
            }
            else if (State == State.DataLoaded)
            {
                Print(string.Format("=== v9 INIT === BarsArray.Length={0} SMTEnabled={1} UseCSV={2}",
                    BarsArray.Length, SMTEnabled, UseCSVSignals));

                atrIndicator = ATR(14);

                candleBuffer = new List<BarData>();
                fvgs5m = new List<FVG>();
                htfFVGs1h = new List<HTF_FVG>();
                htfFVGs4h = new List<HTF_FVG>();

                swingHighs = new List<SwingPoint>();
                swingLows = new List<SwingPoint>();
                htfSwingHighs = new List<SwingPoint>();
                htfSwingLows = new List<SwingPoint>();
                esSwingHighs = new List<SwingPoint>();
                esSwingLows = new List<SwingPoint>();
                esHighBuffer = new List<double>();
                esLowBuffer = new List<double>();

                sweptLowBuffer = new List<KeyValuePair<int, bool>>();
                sweptHighBuffer = new List<KeyValuePair<int, bool>>();

                tradeLog = new List<Dictionary<string, object>>();
                newsEventTimes = new List<DateTime>();
                signalMap = new Dictionary<long, SignalRow>();

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

                // Overnight
                overnightHigh = double.NaN;
                overnightLow = double.NaN;
                overnightRunningH = double.NaN;
                overnightRunningL = double.NaN;
                inOvernight = false;

                // ORM
                ormHigh = double.NaN;
                ormLow = double.NaN;
                ormRunningH = double.NaN;
                ormRunningL = double.NaN;
                inORM = false;
                ormBias = 0;
                ormBiasLocked = false;
                nyOpenPrice = double.NaN;
                nyOpenLocked = false;

                // State
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
                htfBias4h = 0;
                htfBias1h = 0;
                htfPDACount = 0;

                lastNQSwingHighPrice = double.NaN;
                lastNQSwingLowPrice = double.NaN;
                lastESSwingHighPrice = double.NaN;
                lastESSwingLowPrice = double.NaN;
                currentSMTBull = false;
                currentSMTBear = false;

                pendingSignal = null;
                pendingEarlyCut = false;

                rollDates = new HashSet<DateTime>();
                InitRollDates();
                LoadNewsCalendar();

                if (UseCSVSignals)
                    LoadSignalCSV();

                Print(string.Format("  Rollover dates: {0}, News events: {1}, CSV signals: {2}",
                    rollDates.Count, newsEventTimes.Count, signalMap.Count));
            }
            else if (State == State.Terminated)
            {
                if (totalTrades > 0)
                {
                    double wr = (double)totalWins / totalTrades * 100.0;
                    double maxDD = peakR - cumR;
                    if (maxDD < 0) maxDD = 0;
                    double ppdd = maxDD > 0.01 ? totalR / maxDD : 0;
                    Print(string.Format("=== v9 FINAL === Trades={0} WR={1:F1}% TotalR={2:F1} MaxDD={3:F1}R PPDD={4:F1} CumR={5:F1}",
                        totalTrades, wr, totalR, maxDD, ppdd, cumR));
                    Print(string.Format("  Mode={0} Trend={1} MSS={2}", UseCSVSignals ? "CSV" : "Internal", dbgTrendSignals, dbgMSSSignals));
                    Print(string.Format("  Filtered: Session={0} Bias={1} SQ={2} MinStop={3}",
                        dbgFilteredSession, dbgFilteredBias, dbgFilteredSQ, dbgFilteredMinStop));
                }

                // CSV trade log export
                if (tradeLog != null && tradeLog.Count > 0)
                {
                    string csvPath = System.IO.Path.Combine(
                        Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                        "nt8_v9_trades_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".csv");
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

        /// <summary>
        /// Main 5m bar processing. Flow matches bar_by_bar_engine.on_bar() exactly:
        ///   1. barIndex++
        ///   2. ATR (use indicator)
        ///   3. append to candleBuffer
        ///   4. UpdateSwings + UpdateHTFSwings
        ///   5. UpdateSession
        ///   6. DetectNewFVG5m
        ///   7. UpdateSweepTracking
        ///   8. Handle pending early cut
        ///   9. ManagePosition
        ///  10. Handle pending entry
        ///  11. ORM / dayStopped checks
        ///  12-13. CheckTrendSignal + CheckMSSSignal
        ///  14. SMT gate for MSS
        ///  15. Overnight MSS kill
        ///  16. News blackout
        ///  17. PassesFilters
        ///  18. Save as pending entry
        ///  19. UpdateFVGStates5m (AFTER signals)
        /// </summary>
        private void On5mBar()
        {
            barIndex++;
            double av = atrIndicator[0];
            if (av <= 0 || double.IsNaN(av)) return;

            // Bar time in Eastern (NT8 platform timezone must be ET)
            DateTime barOpenET = Time[0].AddMinutes(-BarsPeriod.Value);
            DateTime barTimeET = barOpenET;
            double hf = barOpenET.Hour + barOpenET.Minute / 60.0;
            string session = GetSession(hf);

            // Debug session counters
            if (session == "ny") dbgNYBars++;
            else if (session == "london") dbgLondonBars++;
            else dbgAsiaBars++;

            // Step 1: Daily reset
            NewDayCheck(barTimeET);

            // Step 2: Append to candle buffer (ring buffer, max 20)
            DateTime c3Date = barOpenET.Date;
            bool isRoll = rollDates.Contains(c3Date);
            candleBuffer.Add(new BarData
            {
                Open = Open[0], High = High[0], Low = Low[0], Close = Close[0],
                IsRollDate = isRoll, BarIdx = barIndex
            });
            if (candleBuffer.Count > CANDLE_BUFFER_MAX)
                candleBuffer.RemoveAt(0);

            // Step 3: Update session tracking (Python _update_session)
            UpdateSession(hf, High[0], Low[0], Open[0]);

            // Step 4: Update swings
            UpdateSwings(High[0], Low[0], barIndex, swingHighs, swingLows,
                         Highs[BIP_NQ5M], Lows[BIP_NQ5M], CurrentBars[BIP_NQ5M]);
            UpdateHTFSwings(High[0], Low[0], barIndex);

            // Step 5: Update sweep tracking (for MSS)
            UpdateSweepTracking(High[0], Low[0], barIndex);

            // Step 6: Detect new FVGs (BEFORE signals, Fix #7)
            if (!UseCSVSignals)
                DetectNewFVG5m(av);

            // Compute SMT divergence
            if (SMTEnabled && BarsArray.Length > BIP_ES5M && CurrentBars[BIP_ES5M] > SMTSweepLookback)
                ComputeSMTDivergence(av);

            // Step 7: Handle pending early cut from previous bar (Fix #10)
            if (ts != null && pendingEarlyCut)
            {
                ExitTrade(Open[0], "early_cut_pa");
                pendingEarlyCut = false;
            }

            // Step 8: Manage open position
            if (ts != null)
            {
                ManagePosition(av, hf, barTimeET);
                if (ts != null) goto PostSignals; // still in trade
            }

            // Step 9: Execute pending signal from previous bar (Fix #3)
            if (pendingSignal != null && ts == null)
            {
                pendingSignal["entry_price"] = (double)Open[0];
                double pendEntry = (double)pendingSignal["entry_price"];
                double pendStop = (double)pendingSignal["model_stop"];
                int pendDir = (int)pendingSignal["direction"];

                // Recalculate TP1 with actual entry price (v8.1 fix)
                pendingSignal["tp1"] = FindIRL(pendDir, pendEntry, pendStop);

                bool stopValid = (pendDir == 1 && pendStop < pendEntry) ||
                                 (pendDir == -1 && pendStop > pendEntry);
                if (stopValid)
                {
                    // Check min stop dist with actual entry
                    double stopDist = Math.Abs(pendEntry - pendStop);
                    if (stopDist >= MinStopAtrMult * av)
                    {
                        EnterTrade(pendingSignal, av, session, hf, barTimeET);
                        // FIX: Do NOT mark FVG as "used". Python only uses cooldown
                        // (last_signal_idx), never permanently kills FVGs after one signal.
                        // Removing "used" status allows FVGs to signal again after cooldown.
                    }
                }
                pendingSignal = null;
            }

            if (ts != null) goto PostSignals;

            // Lunch dead zone skip
            if (hf >= LunchStart && hf < LunchEnd) goto PostSignals;

            // ORM no-trade window (9:30-10:00 ET, inclusive of 10:00)
            // FIX: Python skips 9:30-10:00 inclusive (et_h==10 and et_m==0 is also skipped).
            // For 5m bars, bar open 10:00 has hf=10.0 exactly. Must include it.
            if (hf >= 9.5 && hf <= 10.0) goto PostSignals;

            // Step 10: Signal detection
            if (!dayStopped)
            {
                if (UseCSVSignals)
                {
                    // CSV mode: look up signal from previous bar
                    DateTime signalBarTime = barOpenET.AddMinutes(-BarsPeriod.Value);
                    long lookupKey = signalBarTime.Ticks;

                    if (signalMap.ContainsKey(lookupKey) && ts == null)
                    {
                        var sig = signalMap[lookupKey];
                        dbgSignalsFound++;

                        if (consecutiveLosses >= MaxConsecLosses || dailyPnlR <= -DailyMaxLossR)
                            goto PostSignals;
                        if (IsInNewsBlackout(barTimeET))
                            goto PostSignals;

                        double entry = Open[0];
                        double stop = sig.ModelStop;
                        double stopDist = Math.Abs(entry - stop);
                        bool stopValid = (sig.Direction == 1 && stop < entry) ||
                                         (sig.Direction == -1 && stop > entry);
                        if (!stopValid) goto PostSignals;
                        if (stopDist < MinStopAtrMult * av) goto PostSignals;

                        double irl = sig.IRL_Target > 0 ? sig.IRL_Target : FindIRL(sig.Direction, entry, stop);
                        var sigDict = new Dictionary<string, object>
                        {
                            {"direction", sig.Direction}, {"type", sig.Type},
                            {"entry_price", entry}, {"model_stop", stop}, {"tp1", irl},
                            {"has_smt", sig.HasSMT}, {"sweep_score", sig.SweepScore},
                            {"_csv_grade", sig.Grade}
                        };
                        EnterTrade(sigDict, av, session, hf, barTimeET);
                        if (ts != null) dbgSignalsExecuted++;
                    }
                }
                else
                {
                    // Internal mode: detect signals (Python on_bar lines 2290-2336)
                    var sig = CheckSignals(av, session, hf);
                    if (sig != null)
                        pendingSignal = sig;
                }
            }

        PostSignals:
            // Step 11: Update FVG states AFTER signal detection (Fix #7)
            if (!UseCSVSignals)
                UpdateFVGStates5m(av);
        }

        #endregion

        #region HTF Bar Handlers

        /// <summary>
        /// On 1H bar: detect HTF FVGs, update states, recompute bias.
        /// Matches Python on_htf_bar('1H', ...).
        /// </summary>
        private void On1hBar()
        {
            if (CurrentBars[BIP_NQ1H] < 3) return;

            double c1High = Highs[BIP_NQ1H][2], c1Low = Lows[BIP_NQ1H][2];
            double c2Open = Opens[BIP_NQ1H][1], c2High = Highs[BIP_NQ1H][1];
            double c2Low = Lows[BIP_NQ1H][1], c2Close = Closes[BIP_NQ1H][1];
            double c3High = Highs[BIP_NQ1H][0], c3Low = Lows[BIP_NQ1H][0];

            // Detect bullish FVG
            if (c1High < c3Low)
            {
                htfFVGs1h.Add(new HTF_FVG
                {
                    BarIndex = barIndex, Direction = 1,
                    Top = c3Low, Bottom = c1High, Size = c3Low - c1High
                });
            }
            // Detect bearish FVG
            if (c1Low > c3High)
            {
                htfFVGs1h.Add(new HTF_FVG
                {
                    BarIndex = barIndex, Direction = -1,
                    Top = c1Low, Bottom = c3High, Size = c1Low - c3High
                });
            }

            // Update HTF FVG states
            double close1h = Closes[BIP_NQ1H][0];
            double high1h = Highs[BIP_NQ1H][0];
            double low1h = Lows[BIP_NQ1H][0];

            for (int f = htfFVGs1h.Count - 1; f >= 0; f--)
            {
                var fvg = htfFVGs1h[f];
                if (fvg.Status == "invalidated") continue;
                if (fvg.Direction == 1 && close1h < fvg.Bottom)
                    fvg.Status = "invalidated";
                else if (fvg.Direction == -1 && close1h > fvg.Top)
                    fvg.Status = "invalidated";
                if (fvg.Direction == 1 && low1h <= fvg.Top && fvg.Status == "untested")
                    fvg.Status = "tested_rejected";
                else if (fvg.Direction == -1 && high1h >= fvg.Bottom && fvg.Status == "untested")
                    fvg.Status = "tested_rejected";
            }

            // Prune old and invalidated
            htfFVGs1h.RemoveAll(f => f.Status == "invalidated" || barIndex - f.BarIndex > 1200);

            RecomputeHTFBias(Closes[BIP_NQ1H][0]);
        }

        /// <summary>
        /// On 4H bar: detect HTF FVGs, update states, recompute bias.
        /// Matches Python on_htf_bar('4H', ...).
        /// </summary>
        private void On4hBar()
        {
            if (CurrentBars[BIP_NQ4H] < 3) return;

            double c1High = Highs[BIP_NQ4H][2], c1Low = Lows[BIP_NQ4H][2];
            double c2Open = Opens[BIP_NQ4H][1], c2High = Highs[BIP_NQ4H][1];
            double c2Low = Lows[BIP_NQ4H][1], c2Close = Closes[BIP_NQ4H][1];
            double c3High = Highs[BIP_NQ4H][0], c3Low = Lows[BIP_NQ4H][0];

            if (c1High < c3Low)
            {
                htfFVGs4h.Add(new HTF_FVG
                {
                    BarIndex = barIndex, Direction = 1,
                    Top = c3Low, Bottom = c1High, Size = c3Low - c1High
                });
            }
            if (c1Low > c3High)
            {
                htfFVGs4h.Add(new HTF_FVG
                {
                    BarIndex = barIndex, Direction = -1,
                    Top = c1Low, Bottom = c3High, Size = c1Low - c3High
                });
            }

            double close4h = Closes[BIP_NQ4H][0];
            double high4h = Highs[BIP_NQ4H][0];
            double low4h = Lows[BIP_NQ4H][0];

            for (int f = htfFVGs4h.Count - 1; f >= 0; f--)
            {
                var fvg = htfFVGs4h[f];
                if (fvg.Status == "invalidated") continue;
                if (fvg.Direction == 1 && close4h < fvg.Bottom)
                    fvg.Status = "invalidated";
                else if (fvg.Direction == -1 && close4h > fvg.Top)
                    fvg.Status = "invalidated";
                if (fvg.Direction == 1 && low4h <= fvg.Top && fvg.Status == "untested")
                    fvg.Status = "tested_rejected";
                else if (fvg.Direction == -1 && high4h >= fvg.Bottom && fvg.Status == "untested")
                    fvg.Status = "tested_rejected";
            }

            htfFVGs4h.RemoveAll(f => f.Status == "invalidated" || barIndex - f.BarIndex > 4800);

            RecomputeHTFBias(Closes[BIP_NQ4H][0]);
        }

        /// <summary>
        /// On ES 5m bar: update ES swings for SMT divergence.
        /// Matches Python on_es_bar().
        /// </summary>
        private void OnES5mBar()
        {
            if (CurrentBars[BIP_ES5M] < 10) return;

            UpdateSwings(
                Highs[BIP_ES5M][0], Lows[BIP_ES5M][0], barIndex,
                esSwingHighs, esSwingLows,
                Highs[BIP_ES5M], Lows[BIP_ES5M], CurrentBars[BIP_ES5M]);

            esHighBuffer.Add(Highs[BIP_ES5M][0]);
            esLowBuffer.Add(Lows[BIP_ES5M][0]);
            if (esHighBuffer.Count > SMTSweepLookback + 10)
            {
                esHighBuffer.RemoveAt(0);
                esLowBuffer.RemoveAt(0);
            }

            if (esSwingHighs.Count > 0)
                lastESSwingHighPrice = esSwingHighs[esSwingHighs.Count - 1].Price;
            if (esSwingLows.Count > 0)
                lastESSwingLowPrice = esSwingLows[esSwingLows.Count - 1].Price;
        }

        #endregion

        #region FVG Detection & State Machine

        /// <summary>
        /// DetectNewFVG5m: birth detection only.
        /// Matches Python _detect_fvg() exactly.
        /// c1=buffer[-3], c2=buffer[-2], c3=buffer[-1].
        /// Rollover filter: skip FVGs if any of c1/c2/c3 is_roll_date.
        /// Sweep scoring: session levels + HTF swing levels.
        /// </summary>
        private void DetectNewFVG5m(double av)
        {
            if (candleBuffer.Count < 3) return;

            int n = candleBuffer.Count;
            BarData c1 = candleBuffer[n - 3];
            BarData c2 = candleBuffer[n - 2];
            BarData c3 = candleBuffer[n - 1];

            // Rollover filter
            if (c1.IsRollDate || c2.IsRollDate || c3.IsRollDate)
                return;

            // Sweep scoring from c2 (displacement candle)
            int sweepScore = ComputeSweepScore(c2.High, c2.Low);
            bool swept = sweepScore >= 2;

            // Bullish FVG: c1.High < c3.Low
            if (c1.High < c3.Low)
            {
                double sz = c3.Low - c1.High;
                fvgs5m.Add(new FVG
                {
                    BarIndex = barIndex,
                    Direction = 1,
                    Top = c3.Low,
                    Bottom = c1.High,
                    Size = sz,
                    Candle2Open = c2.Open,
                    CreationTime = Time[0],
                    SweptLiquidity = swept,
                    SweepScore = sweepScore
                });
            }

            // Bearish FVG: c1.Low > c3.High
            if (c1.Low > c3.High)
            {
                double sz = c1.Low - c3.High;
                fvgs5m.Add(new FVG
                {
                    BarIndex = barIndex,
                    Direction = -1,
                    Top = c1.Low,
                    Bottom = c3.High,
                    Size = sz,
                    Candle2Open = c2.Open,
                    CreationTime = Time[0],
                    SweptLiquidity = swept,
                    SweepScore = sweepScore
                });
            }
        }

        /// <summary>
        /// UpdateFVGStates5m: update existing FVG statuses, spawn IFVGs, prune.
        /// Matches Python _update_all_fvg_states() exactly.
        /// Called AFTER signal detection (Fix #7).
        /// </summary>
        private void UpdateFVGStates5m(double av)
        {
            double curHigh = High[0], curLow = Low[0], curClose = Close[0];
            var newIFVGs = new List<FVG>();

            for (int f = fvgs5m.Count - 1; f >= 0; f--)
            {
                var fvg = fvgs5m[f];
                // FIX: Remove "used" skip. Python never uses "used" status.
                // FVGs that have signaled must still be invalidated/spawned as IFVGs.
                if (fvg.Status == "invalidated") continue;

                int effectiveDir = fvg.IsIFVG ? fvg.IFVGDirection : fvg.Direction;

                if (effectiveDir == 1) // Bull
                {
                    if (curLow <= fvg.Top && fvg.Status == "untested")
                        fvg.Status = "tested_rejected";

                    if (curClose < fvg.Bottom)
                    {
                        fvg.Status = "invalidated";
                        fvg.InvalidatedAtIdx = barIndex;
                        if (!fvg.IsIFVG)
                        {
                            // Spawn Bear IFVG (bull invalidated -> resistance)
                            newIFVGs.Add(new FVG
                            {
                                BarIndex = barIndex,
                                Direction = 1,  // original direction
                                Top = fvg.Top,
                                Bottom = fvg.Bottom,
                                Size = fvg.Size,
                                Candle2Open = fvg.Top,  // stop = top for bear IFVG
                                CreationTime = fvg.CreationTime,
                                IsIFVG = true,
                                IFVGDirection = -1,  // inverted
                                SweepScore = fvg.SweepScore,
                                SweptLiquidity = fvg.SweptLiquidity
                            });
                        }
                    }
                }
                else // Bear
                {
                    if (curHigh >= fvg.Bottom && fvg.Status == "untested")
                        fvg.Status = "tested_rejected";

                    if (curClose > fvg.Top)
                    {
                        fvg.Status = "invalidated";
                        fvg.InvalidatedAtIdx = barIndex;
                        if (!fvg.IsIFVG)
                        {
                            // Spawn Bull IFVG (bear invalidated -> support)
                            newIFVGs.Add(new FVG
                            {
                                BarIndex = barIndex,
                                Direction = -1,  // original direction
                                Top = fvg.Top,
                                Bottom = fvg.Bottom,
                                Size = fvg.Size,
                                Candle2Open = fvg.Bottom,  // stop = bottom for bull IFVG
                                CreationTime = fvg.CreationTime,
                                IsIFVG = true,
                                IFVGDirection = 1,  // inverted
                                SweepScore = fvg.SweepScore,
                                SweptLiquidity = fvg.SweptLiquidity
                            });
                        }
                    }
                }
            }

            // Add new IFVGs
            fvgs5m.AddRange(newIFVGs);

            // Prune old FVGs (>500 bars)
            fvgs5m.RemoveAll(f => barIndex - f.BarIndex > 500);
        }

        /// <summary>
        /// Compute sweep score from c2's high/low against completed session levels and HTF swings.
        /// Matches Python _detect_fvg() sweep logic.
        /// </summary>
        private int ComputeSweepScore(double candleHigh, double candleLow)
        {
            int score = 0;

            // Session levels (completed sub-session H/L)
            for (int s = 0; s < 5; s++)
            {
                if (!double.IsNaN(sessCompletedH[s]) && candleHigh > sessCompletedH[s])
                    score += 2;
                if (!double.IsNaN(sessCompletedL[s]) && candleLow < sessCompletedL[s])
                    score += 2;
            }

            // HTF swing levels
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

        #endregion

        #region Signal Detection

        /// <summary>
        /// Check for signals: Trend first, then MSS.
        /// Returns pending signal dict or null.
        /// Matches Python on_bar() signal detection flow.
        /// </summary>
        private Dictionary<string, object> CheckSignals(double av, string session, double hf)
        {
            if (CurrentBars[BIP_NQ5M] < 10) return null;

            Dictionary<string, object> signal = null;

            // Try Trend signal
            signal = CheckTrendSignal(av);
            if (signal != null)
            {
                dbgTrendSignals++;
                return ApplySignalFilters(signal, av, session, hf);
            }

            // Try MSS signal
            var mssSignal = CheckMSSSignal(av);
            if (mssSignal != null)
            {
                dbgMSSSignals++;

                // SMT gate for MSS (Python lines 2302-2312)
                int mssDir = (int)mssSignal["direction"];
                if (SMTRequireForMSS && SMTEnabled)
                {
                    if ((mssDir == 1 && currentSMTBull) || (mssDir == -1 && currentSMTBear))
                        signal = mssSignal;
                    // else: blocked by SMT gate
                }
                else if (!SMTRequireForMSS)
                {
                    signal = mssSignal;
                }

                // Overnight MSS kill (16:00-03:00 ET)
                if (signal != null && (string)signal["type"] == "mss")
                {
                    DateTime barOpenET = Time[0].AddMinutes(-BarsPeriod.Value);
                    double etFrac = barOpenET.Hour + barOpenET.Minute / 60.0;
                    if (etFrac >= 16.0 || etFrac < 3.0)
                        signal = null; // kill
                }

                if (signal != null)
                    return ApplySignalFilters(signal, av, session, hf);
            }

            return null;
        }

        /// <summary>
        /// Apply filter chain to a signal. If passes, return as pending entry.
        /// Matches Python _passes_filters() -> save as _pending_entry.
        /// </summary>
        private Dictionary<string, object> ApplySignalFilters(Dictionary<string, object> signal,
            double av, string session, double hf)
        {
            DateTime barOpenET = Time[0].AddMinutes(-BarsPeriod.Value);
            DateTime barTimeET = barOpenET;

            // News blackout pre-check
            if (IsInNewsBlackout(barTimeET))
                return null;

            if (!PassesFilters(signal, av, session, hf, barTimeET))
                return null;

            return signal;
        }

        /// <summary>
        /// CheckTrendSignal: FVG test + rejection.
        /// Matches Python _check_trend_signal() exactly.
        /// </summary>
        private Dictionary<string, object> CheckTrendSignal(double av)
        {
            double curClose = Close[0], curOpen = Open[0];
            double curHigh = High[0], curLow = Low[0];

            double bd = Math.Abs(curClose - curOpen);
            double rn = curHigh - curLow;
            if (rn <= 0 || bd / rn < FVGRejectionBodyRatio) return null;

            // Fluency pre-check
            if (!CheckFluency(av)) return null;

            // Displacement check
            bool curDisplaced = CheckDisplacement(curOpen, curHigh, curLow, curClose, av);

            int bestDir = 0;
            FVG bestFVG = null;
            double bestScore = -1;

            foreach (var f in fvgs5m)
            {
                // FIX: Remove "used" check. Python never sets status="used",
                // it only uses cooldown (last_signal_idx). FVGs can signal
                // multiple times, gated only by cooldown.
                if (f.Status == "invalidated") continue;
                if (f.IsIFVG) continue;
                if (f.BarIndex >= barIndex) continue;
                if (barIndex - f.LastSignalIdx < FVGCooldown) continue;
                if (av > 0 && f.Size < FVGMinSizeAtr * av) continue;

                if (f.Direction == 1) // Bull FVG
                {
                    bool entered = curLow <= f.Top && curHigh >= f.Bottom;
                    bool rejected = curClose > f.Top;
                    if (entered && rejected)
                    {
                        double score = f.Size + (curDisplaced ? 100.0 : 0.0)
                                       + (f.SweptLiquidity ? 200.0 : 0.0);
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
                        double score = f.Size + (curDisplaced ? 100.0 : 0.0)
                                       + (f.SweptLiquidity ? 200.0 : 0.0);
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

            bestFVG.LastSignalIdx = barIndex;
            double entry = curClose; // placeholder, replaced by next bar open
            double stop = bestFVG.Candle2Open;
            double tp1 = FindIRL(bestDir, entry, stop);

            return new Dictionary<string, object>
            {
                {"direction", bestDir}, {"type", "trend"}, {"entry_price", entry},
                {"model_stop", stop}, {"tp1", tp1}, {"has_smt", false},
                {"sweep_score", bestFVG.SweepScore},
                {"_fvg_index", fvgs5m.IndexOf(bestFVG)}
            };
        }

        /// <summary>
        /// CheckMSSSignal: IFVG retest + respect.
        /// Matches Python _check_mss_signal() exactly.
        /// </summary>
        private Dictionary<string, object> CheckMSSSignal(double av)
        {
            double curClose = Close[0], curOpen = Open[0];
            double curHigh = High[0], curLow = Low[0];

            double bd = Math.Abs(curClose - curOpen);
            double rn = curHigh - curLow;
            if (rn <= 0 || bd / rn < FVGRejectionBodyRatio) return null;
            if (!CheckFluency(av)) return null;

            bool curDisplaced = CheckDisplacement(curOpen, curHigh, curLow, curClose, av);

            int bestDir = 0;
            FVG bestIFVG = null;
            double bestScore = -1;

            foreach (var f in fvgs5m)
            {
                if (!f.IsIFVG) continue;
                // FIX: Remove "used" check (same as trend signal fix above)
                if (f.Status == "invalidated") continue;
                if (f.BarIndex >= barIndex) continue;
                if (barIndex - f.LastSignalIdx < FVGCooldown) continue;
                if (av > 0 && f.Size < FVGMinSizeAtr * av) continue;

                int idir = f.IFVGDirection;

                // Sweep check before IFVG birth (Python _had_sweep_low_before / _had_sweep_high_before)
                bool hadSweep;
                if (idir == 1) // Bull IFVG: need prior swing low sweep
                    hadSweep = HadSweepLowBefore(f.BarIndex);
                else // Bear IFVG: need prior swing high sweep
                    hadSweep = HadSweepHighBefore(f.BarIndex);

                if (!hadSweep) continue;

                if (idir == 1) // Bull IFVG
                {
                    bool entered = curLow <= f.Top && curHigh >= f.Bottom;
                    bool respected = curClose > f.Top;
                    if (entered && respected)
                    {
                        double score = f.Size + (curDisplaced ? 100.0 : 0.0);
                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestDir = 1;
                            bestIFVG = f;
                        }
                    }
                }
                else // Bear IFVG
                {
                    bool entered = curHigh >= f.Bottom && curLow <= f.Top;
                    bool respected = curClose < f.Bottom;
                    if (entered && respected)
                    {
                        double score = f.Size + (curDisplaced ? 100.0 : 0.0);
                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestDir = -1;
                            bestIFVG = f;
                        }
                    }
                }
            }

            if (bestIFVG == null) return null;

            bestIFVG.LastSignalIdx = barIndex;
            double entry = Close[0];
            double stop = bestDir == 1 ? bestIFVG.Bottom : bestIFVG.Top;
            double tp1 = FindIRL(bestDir, entry, stop);

            // SMT gating is done in CheckSignals, not here
            bool hasSMT = false;
            if (SMTEnabled)
                hasSMT = (bestDir == 1 && currentSMTBull) || (bestDir == -1 && currentSMTBear);

            return new Dictionary<string, object>
            {
                {"direction", bestDir}, {"type", "mss"}, {"entry_price", entry},
                {"model_stop", stop}, {"tp1", tp1}, {"has_smt", hasSMT},
                {"sweep_score", bestIFVG.SweepScore},
                {"_fvg_index", fvgs5m.IndexOf(bestIFVG)}
            };
        }

        #endregion

        #region Displacement & Fluency

        /// <summary>
        /// CheckDisplacement: body > atr_mult * ATR, body/range >= body_ratio, engulfs >= 1 prior.
        /// Matches Python _check_displacement().
        /// </summary>
        private bool CheckDisplacement(double open_, double high, double low, double close, double atr)
        {
            if (atr <= 0) return false;
            double body = Math.Abs(close - open_);
            double range = high - low;
            if (range <= 0) return false;

            if (body <= DispAtrMult * atr) return false;
            if (body / range < DispBodyRatio) return false;

            // Engulfment check — FIX: Python only checks engulf_min_candles (=1)
            // prior bars, not up to 3. Using 1 to match Python exactly.
            if (CurrentBars[BIP_NQ5M] < 2) return false;
            double bodyHigh = Math.Max(close, open_);
            double bodyLow = Math.Min(close, open_);
            bool engulfs = false;
            // Only check 1 prior bar (engulf_min_candles=1 from params.yaml)
            int engulfLookback = 1;  // matches Python engulf_min_candles
            for (int k = 1; k <= Math.Min(engulfLookback, CurrentBars[BIP_NQ5M] - 1); k++)
            {
                if (bodyHigh >= High[k] && bodyLow <= Low[k])
                { engulfs = true; break; }
            }
            return engulfs;
        }

        /// <summary>
        /// CheckFluency: max(bull,bear)/window formula.
        /// Matches Python _compute_fluency() with threshold check.
        /// </summary>
        private bool CheckFluency(double av)
        {
            double flu = ComputeFluency(av);
            return !double.IsNaN(flu) && flu >= FluencyThreshold;
        }

        /// <summary>
        /// ComputeFluency: rolling fluency score over last N bars from candle buffer.
        /// Matches Python _compute_fluency() exactly: max(bull_count, bear_count) / window.
        /// </summary>
        private double ComputeFluency(double av)
        {
            int window = FluencyWindow;
            if (candleBuffer.Count < window) return double.NaN;

            int bullCount = 0, bearCount = 0;
            double sumBodyRatio = 0;
            double sumBarSize = 0;

            int start = candleBuffer.Count - window;
            for (int j = start; j < candleBuffer.Count; j++)
            {
                var bar = candleBuffer[j];
                double body = Math.Abs(bar.Close - bar.Open);
                double range = bar.High - bar.Low;

                if (bar.Close > bar.Open) bullCount++;
                else if (bar.Close < bar.Open) bearCount++;

                if (range > 0)
                    sumBodyRatio += body / range;

                if (av > 0)
                    sumBarSize += Math.Min(range / av, 2.0);
            }

            double dirRatio = (double)Math.Max(bullCount, bearCount) / window;
            double avgBodyRatio = sumBodyRatio / window;
            double avgBarSize = av > 0 ? Math.Min(sumBarSize / window, 1.0) : 0.0;

            double fluency = FluencyWDir * dirRatio + FluencyWBody * avgBodyRatio
                           + FluencyWBar * avgBarSize;
            return Math.Max(0.0, Math.Min(1.0, fluency));
        }

        /// <summary>
        /// ComputeFluencyScore for signal quality (returns raw score, not bool).
        /// </summary>
        private double ComputeFluencyScore(double av)
        {
            return ComputeFluency(av);
        }

        #endregion

        #region Signal Quality & PA

        /// <summary>
        /// ComputeSignalQuality: composite score from size, displacement, fluency, PA.
        /// Matches Python _compute_signal_quality().
        /// </summary>
        private double ComputeSignalQuality(Dictionary<string, object> sig, double av)
        {
            double entry = (double)sig["entry_price"];
            double stop = (double)sig["model_stop"];

            double gap = Math.Abs(entry - stop);
            double sizeSc = av > 0 ? Math.Min(1.0, gap / (av * 1.5)) : 0.5;

            double body = Math.Abs(Close[0] - Open[0]);
            double range = High[0] - Low[0];
            double dispSc = range > 0 ? body / range : 0;

            double fluSc = ComputeFluencyScore(av);
            if (double.IsNaN(fluSc)) fluSc = 0.5;
            fluSc = Math.Min(1.0, Math.Max(0.0, fluSc));

            double paSc = ComputePAScore();

            return SQWSize * sizeSc + SQWDisp * dispSc + SQWFlu * fluSc + SQWPA * paSc;
        }

        /// <summary>
        /// ComputePAScore: 1 - alt_dir_ratio over 6 bars BEFORE current bar.
        /// FIX: Python export uses c[idx-6:idx] which EXCLUDES the current bar.
        /// v9 was using candleBuffer[-6:] which INCLUDES current bar (off-by-one).
        /// Now uses candleBuffer[-7:-1] to match Python export.
        /// </summary>
        private double ComputePAScore()
        {
            int window = 6;
            // Need window+1 bars: 6 bars before current + current bar itself
            if (candleBuffer.Count < window + 1) return 0.5;

            int altCount = 0;
            // Start at candleBuffer.Count - window - 1, end at candleBuffer.Count - 1 (exclusive)
            // This gives 6 bars BEFORE the current bar
            int start = candleBuffer.Count - window - 1;
            int end = candleBuffer.Count - 1; // exclusive (skip current bar)
            for (int j = start + 1; j < end; j++)
            {
                var prev = candleBuffer[j - 1];
                var cur = candleBuffer[j];
                int dir1 = prev.Close > prev.Open ? 1 : (prev.Close < prev.Open ? -1 : 0);
                int dir2 = cur.Close > cur.Open ? 1 : (cur.Close < cur.Open ? -1 : 0);
                if (dir1 != dir2) altCount++;
            }
            return 1.0 - (double)altCount / (window - 1);
        }

        /// <summary>
        /// ComputeAltDirRatio: alternating direction ratio for PA quality filter.
        /// Matches Python _compute_alt_dir_ratio().
        /// </summary>
        private double ComputeAltDirRatio()
        {
            int window = 6;
            if (candleBuffer.Count < window) return 0.0;

            int altCount = 0;
            int start = candleBuffer.Count - window;
            for (int j = start + 1; j < candleBuffer.Count; j++)
            {
                var prev = candleBuffer[j - 1];
                var cur = candleBuffer[j];
                int dir1 = prev.Close > prev.Open ? 1 : (prev.Close < prev.Open ? -1 : 0);
                int dir2 = cur.Close > cur.Open ? 1 : (cur.Close < cur.Open ? -1 : 0);
                if (dir1 != dir2) altCount++;
            }
            return (double)altCount / (window - 1);
        }

        #endregion

        #region Bias Computation

        /// <summary>
        /// RecomputeHTFBias: count active FVGs above/below price.
        /// Matches Python _recompute_htf_bias().
        /// </summary>
        private void RecomputeHTFBias(double currentClose)
        {
            // 4H draw
            int bull4hAbove = 0, bear4hBelow = 0;
            foreach (var f in htfFVGs4h)
            {
                if (f.Status == "invalidated") continue;
                double mid = (f.Top + f.Bottom) / 2.0;
                if (f.Direction == 1 && mid > currentClose) bull4hAbove++;
                else if (f.Direction == -1 && mid < currentClose) bear4hBelow++;
            }

            double draw4h = 0;
            if (bull4hAbove > 0 && bear4hBelow == 0) draw4h = 1.0;
            else if (bear4hBelow > 0 && bull4hAbove == 0) draw4h = -1.0;
            else if (bull4hAbove > 0 && bear4hBelow > 0)
            {
                int net = bull4hAbove - bear4hBelow;
                draw4h = net > 0 ? 0.5 : (net < 0 ? -0.5 : 0.0);
            }

            // 1H draw
            int bull1hAbove = 0, bear1hBelow = 0;
            foreach (var f in htfFVGs1h)
            {
                if (f.Status == "invalidated") continue;
                double mid = (f.Top + f.Bottom) / 2.0;
                if (f.Direction == 1 && mid > currentClose) bull1hAbove++;
                else if (f.Direction == -1 && mid < currentClose) bear1hBelow++;
            }

            double draw1h = 0;
            if (bull1hAbove > 0 && bear1hBelow == 0) draw1h = 1.0;
            else if (bear1hBelow > 0 && bull1hAbove == 0) draw1h = -1.0;
            else if (bull1hAbove > 0 && bear1hBelow > 0)
            {
                int net = bull1hAbove - bear1hBelow;
                draw1h = net > 0 ? 0.5 : (net < 0 ? -0.5 : 0.0);
            }

            htfBias4h = draw4h;
            htfBias1h = draw1h;
            // FIX: Python counts ALL active HTF FVGs for PDA count, not just
            // those on the "correct side" of price. Use total count.
            int total4h = 0, total1h = 0;
            foreach (var f in htfFVGs4h) { if (f.Status != "invalidated") total4h++; }
            foreach (var f in htfFVGs1h) { if (f.Status != "invalidated") total1h++; }
            htfPDACount = total4h + total1h;
        }

        /// <summary>
        /// GetCompositeBias: 0.4*HTF + 0.3*overnight + 0.3*ORM.
        /// Matches Python _get_composite_bias().
        /// Returns composite bias direction: +1, -1, or 0.
        /// </summary>
        private int GetCompositeBias()
        {
            double htfBias = 0.6 * htfBias4h + 0.4 * htfBias1h;

            // Overnight bias: position of NY open in overnight range
            double overnightBias = 0;
            if (!double.IsNaN(nyOpenPrice) && !double.IsNaN(overnightHigh) && !double.IsNaN(overnightLow))
            {
                double range = overnightHigh - overnightLow;
                if (range > 0)
                {
                    double pos = (nyOpenPrice - overnightLow) / range;
                    if (pos > 0.6) overnightBias = 1.0;
                    else if (pos < 0.4) overnightBias = -1.0;
                }
            }

            double composite = 0.4 * htfBias + 0.3 * overnightBias + 0.3 * ormBias;
            if (Math.Abs(composite) > 0.2)
                return composite > 0 ? 1 : -1;
            return 0;
        }

        /// <summary>
        /// ComputeGrade: A+ / B+ / C based on bias alignment and regime.
        /// FIX: Python's regime uses abs(htf_bias) > 0.2, not composite bias.
        /// The regime should reflect HTF FVG draw strength, not overnight/ORM.
        /// </summary>
        private string ComputeGrade(int dir)
        {
            int biasCur = GetCompositeBias();
            bool aligned = (dir == Math.Sign(biasCur) && biasCur != 0);

            // FIX: Regime based on HTF bias strength (not composite)
            // Python: regime = 1.0 if htf_pda_count > 0 and abs(htf_bias) > 0.2
            //         regime = 0.5 if htf_pda_count > 0
            //         regime = 0.0 otherwise
            double htfBias = 0.6 * htfBias4h + 0.4 * htfBias1h;
            double regime;
            if (htfPDACount > 0 && Math.Abs(htfBias) > 0.2) regime = 1.0;
            else if (htfPDACount > 0) regime = 0.5;
            else regime = 0.0;

            if (regime == 0.0) return "C";
            if (aligned && regime >= 1.0) return "A+";
            if (aligned || regime >= 1.0) return "B+";
            return "C";
        }

        #endregion

        #region SMT Divergence

        /// <summary>
        /// ComputeSMTDivergence: NQ swept but ES didn't.
        /// Matches Python on_es_bar() SMT logic.
        /// </summary>
        private void ComputeSMTDivergence(double av)
        {
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

            // NQ sweep detection
            bool nqSweptHigh = false, nqSweptLow = false;
            int lb = Math.Min(SMTSweepLookback, CurrentBars[BIP_NQ5M]);
            for (int j = 0; j < lb; j++)
            {
                if (High[j] > lastNQSwingHighPrice) nqSweptHigh = true;
                if (Low[j] < lastNQSwingLowPrice) nqSweptLow = true;
            }

            // ES sweep detection
            bool esSweptHigh = false, esSweptLow = false;
            int esLB = Math.Min(SMTSweepLookback + SMTTimeTolerance, esHighBuffer.Count);
            for (int j = esHighBuffer.Count - esLB; j < esHighBuffer.Count; j++)
            {
                if (j < 0) continue;
                if (esHighBuffer[j] > lastESSwingHighPrice) esSweptHigh = true;
                if (esLowBuffer[j] < lastESSwingLowPrice) esSweptLow = true;
            }

            currentSMTBull = nqSweptLow && !esSweptLow;
            currentSMTBear = nqSweptHigh && !esSweptHigh;
        }

        #endregion

        #region Sweep Tracking (for MSS)

        /// <summary>
        /// Track whether current bar sweeps a swing level.
        /// Matches Python _update_sweep_tracking().
        /// </summary>
        private void UpdateSweepTracking(double high, double low, int bi)
        {
            bool sweptLow = false, sweptHigh = false;
            if (swingLows.Count > 0 && !double.IsNaN(swingLows[swingLows.Count - 1].Price))
            {
                if (low < swingLows[swingLows.Count - 1].Price)
                    sweptLow = true;
            }
            if (swingHighs.Count > 0 && !double.IsNaN(swingHighs[swingHighs.Count - 1].Price))
            {
                if (high > swingHighs[swingHighs.Count - 1].Price)
                    sweptHigh = true;
            }

            sweptLowBuffer.Add(new KeyValuePair<int, bool>(bi, sweptLow));
            sweptHighBuffer.Add(new KeyValuePair<int, bool>(bi, sweptHigh));

            if (sweptLowBuffer.Count > SWEPT_BUFFER_MAX)
                sweptLowBuffer.RemoveAt(0);
            if (sweptHighBuffer.Count > SWEPT_BUFFER_MAX)
                sweptHighBuffer.RemoveAt(0);
        }

        /// <summary>
        /// Check if a swing low was swept in lookback window before IFVG birth.
        /// Matches Python _had_sweep_low_before().
        /// </summary>
        private bool HadSweepLowBefore(int ifvgIdx)
        {
            int lookStart = Math.Max(0, ifvgIdx - SweepLookback);
            foreach (var kv in sweptLowBuffer)
            {
                if (kv.Key >= lookStart && kv.Key <= ifvgIdx && kv.Value)
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Check if a swing high was swept in lookback window before IFVG birth.
        /// Matches Python _had_sweep_high_before().
        /// </summary>
        private bool HadSweepHighBefore(int ifvgIdx)
        {
            int lookStart = Math.Max(0, ifvgIdx - SweepLookback);
            foreach (var kv in sweptHighBuffer)
            {
                if (kv.Key >= lookStart && kv.Key <= ifvgIdx && kv.Value)
                    return true;
            }
            return false;
        }

        #endregion

        #region Filters

        /// <summary>
        /// PassesFilters: apply all entry filters.
        /// Matches Python _passes_filters() filter chain.
        /// </summary>
        private bool PassesFilters(Dictionary<string, object> sig, double av,
            string session, double hf, DateTime barTimeET)
        {
            int dir = (int)sig["direction"];
            double entry = (double)sig["entry_price"];
            double stop = (double)sig["model_stop"];
            string sigType = (string)sig["type"];
            bool hasSMT = sig.ContainsKey("has_smt") && (bool)sig["has_smt"];
            bool isMSSSMT = sigType == "mss" && hasSMT && SMTEnabled;

            // FILTER 1: Bias opposing
            int biasCur = GetCompositeBias();
            if (biasCur != 0 && dir == -Math.Sign(biasCur))
            {
                if (!isMSSSMT)
                { dbgFilteredBias++; return false; }
            }

            // FILTER 2: PA quality
            double altDir = ComputeAltDirRatio();
            if (altDir >= PAThreshold)
                return false;

            // FILTER 3: Session filter
            double etFrac = barTimeET.Hour + barTimeET.Minute / 60.0;
            bool mssSessionBypass = isMSSSMT && SMTBypassSession;

            if (!mssSessionBypass)
            {
                if (session == "asia" && SkipAsia) { dbgFilteredSession++; return false; }
                if (session == "london" && SkipLondon) { dbgFilteredSession++; return false; }
                // Default: allow London and Asia only if not skipped
                if (session != "ny" && session != "london" && session != "asia")
                { dbgFilteredSession++; return false; }
            }

            // FILTER 4: Validate stop/entry
            bool stopValid = (dir == 1 && stop < entry) || (dir == -1 && stop > entry);
            if (!stopValid) return false;

            // FILTER 5: Min stop distance
            double stopDist = Math.Abs(entry - stop);
            if (stopDist < MinStopAtrMult * av) { dbgFilteredMinStop++; return false; }

            // FILTER 6: Signal quality
            if (SQEnabled)
            {
                double sq = ComputeSignalQuality(sig, av);
                double threshold = dir == 1 ? SQLongThreshold : SQShortThreshold;
                if (sq < threshold) { dbgFilteredSQ++; return false; }
            }

            // FILTER 7: News blackout
            if (IsInNewsBlackout(barTimeET)) return false;

            // FILTER 8: Daily limits
            if (consecutiveLosses >= MaxConsecLosses) return false;
            if (dailyPnlR <= -DailyMaxLossR) return false;

            // FILTER 9: Already in position
            if (ts != null) return false;

            return true;
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

            // Grade: CSV-provided or computed
            string grade = sig.ContainsKey("_csv_grade") ? (string)sig["_csv_grade"] : ComputeGrade(dir);

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

            // Submit order
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

            // Early PA cut (Fix #10: exit at next bar's open)
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
                double effectiveStop = ts.StopPrice;
                if (ts.Trimmed)
                {
                    effectiveStop = ts.TrailStop > 0 ? ts.TrailStop : ts.StopPrice;
                    if (ts.BeStop > 0 && ts.BeStop > effectiveStop)
                        effectiveStop = ts.BeStop;
                }

                if (Low[0] <= effectiveStop)
                {
                    double exitP = effectiveStop - SlippageTicks * 0.25;
                    string reason = (ts.Trimmed && effectiveStop >= ts.EntryPrice) ? "be_sweep" : "stop";
                    ExitTrade(exitP, reason);
                    return;
                }

                if (!ts.Trimmed && High[0] >= ts.TP1Price)
                {
                    TrimPosition();
                    if (ts == null) return;
                }

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

            double tpDist = Math.Abs(ts.TP1Price - ts.EntryPrice);
            double trimRatio = (double)trimQty / ts.OrigContracts;
            ts.TrimR = (ts.OrigStopDist > 0) ? (tpDist / ts.OrigStopDist) * trimRatio : 0;

            if (fullExit)
            {
                ExitTrade(ts.TP1Price, "tp1");
                return;
            }

            if (ts.Direction == 1)
                ExitLong(trimQty, "TrimLong", "LantoLong");
            else
                ExitShort(trimQty, "TrimShort", "LantoShort");

            ts.Contracts -= trimQty;
            ts.Trimmed = true;
            ts.BeStop = ts.EntryPrice;

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

            double disp = ts.Direction == 1 ? Close[0] - ts.EntryPrice : ts.EntryPrice - Close[0];
            bool noProgress = disp < av * EarlyCutPAThreshold;

            return avgWick > EarlyCutWickRatio && favorable < 0.5 && noProgress;
        }

        private void ExitTrade(double exitPrice, string reason)
        {
            if (ts == null) return;

            if (ts.Direction == 1)
                ExitLong(ts.Contracts, "ExitLong", "LantoLong");
            else
                ExitShort(ts.Contracts, "ExitShort", "LantoShort");

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

            if (reason == "be_sweep" && ts.Trimmed)
            {
                // BE sweep after trim is profitable -- NOT a loss
            }
            else if (rm < 0)
            {
                consecutiveLosses++;
            }
            else
            {
                consecutiveLosses = 0;
            }

            if (consecutiveLosses >= MaxConsecLosses) dayStopped = true;
            if (dailyPnlR <= -DailyMaxLossR) dayStopped = true;

            Print(string.Format("[{0}] d={1} type={2} e={3:F1} x={4:F1} R={5:F2} cum={6:F1} grade={7}",
                reason, ts.Direction, ts.SignalType, ts.EntryPrice, exitPrice, rm, cumR, ts.Grade));

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

        #region Session Tracking

        /// <summary>
        /// UpdateSession: track 5 sub-sessions, overnight, ORM, NY open.
        /// Matches Python _update_session() exactly.
        /// </summary>
        private void UpdateSession(double hf, double barHigh, double barLow, double barOpen)
        {
            // Sub-sessions
            double[][] bounds = new double[][]
            {
                new double[] { 18.0, 3.0 },  // Asia (wraps midnight)
                new double[] { 3.0, 9.5 },   // London
                new double[] { 9.5, 11.0 },  // NY AM
                new double[] { 11.0, 13.0 }, // NY Lunch
                new double[] { 13.0, 16.0 }  // NY PM
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
                        sessCompletedH[s] = sessRunningH[s];
                        sessCompletedL[s] = sessRunningL[s];
                        sessActive[s] = false;
                    }
                }
            }

            // Overnight (18:00-09:30 ET)
            bool nowOvernight = hf >= 18.0 || hf < 9.5;
            if (nowOvernight)
            {
                if (!inOvernight)
                {
                    overnightRunningH = barHigh;
                    overnightRunningL = barLow;
                    inOvernight = true;
                }
                else
                {
                    if (barHigh > overnightRunningH) overnightRunningH = barHigh;
                    if (barLow < overnightRunningL) overnightRunningL = barLow;
                }
            }
            else
            {
                if (inOvernight)
                {
                    overnightHigh = overnightRunningH;
                    overnightLow = overnightRunningL;
                    inOvernight = false;
                }
            }

            // NY open price (first bar at 9:30 ET)
            if (!nyOpenLocked && hf >= 9.5 && hf < 9.5 + 5.0 / 60.0)
            {
                nyOpenPrice = barOpen;
                nyOpenLocked = true;
            }

            // ORM (9:30-10:00 ET)
            bool nowORM = hf >= 9.5 && hf < 10.0;
            if (nowORM)
            {
                if (!inORM)
                {
                    ormRunningH = barHigh;
                    ormRunningL = barLow;
                    inORM = true;
                }
                else
                {
                    if (barHigh > ormRunningH) ormRunningH = barHigh;
                    if (barLow < ormRunningL) ormRunningL = barLow;
                }
            }
            else
            {
                if (inORM)
                {
                    ormHigh = ormRunningH;
                    ormLow = ormRunningL;
                    inORM = false;
                }
            }

            // Lock ORM bias at 10:00 ET
            if (!ormBiasLocked && hf >= 10.0 && !double.IsNaN(overnightHigh))
            {
                if (!double.IsNaN(ormHigh) && !double.IsNaN(overnightHigh))
                {
                    if (ormHigh > overnightHigh)
                        ormBias = 1.0;
                    else if (ormLow < overnightLow)
                        ormBias = -1.0;
                    else
                        ormBias = 0.0;
                    ormBiasLocked = true;
                }
            }

            // Reset at 18:00 ET (new futures session)
            if (hf >= 18.0 && hf < 18.0 + 5.0 / 60.0)
            {
                if (nyOpenLocked)
                {
                    nyOpenLocked = false;
                    nyOpenPrice = double.NaN;
                    ormBiasLocked = false;
                    ormBias = 0;
                    ormHigh = double.NaN;
                    ormLow = double.NaN;
                }
            }
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
                pendingSignal = null;
                pendingEarlyCut = false;
            }
        }

        /// <summary>
        /// Fractal swing detection. Matches Python _update_swings() / _check_swing_at().
        /// Uses NT8 ISeries for bar access.
        /// </summary>
        private void UpdateSwings(double high, double low, int idx,
            List<SwingPoint> shList, List<SwingPoint> slList,
            ISeries<double> highSeries, ISeries<double> lowSeries, int currentBar)
        {
            int right = SwingRightBars;
            int left = SwingLeftBars;
            if (currentBar < left + right + 1) return;

            double candidateHigh = highSeries[right];
            double candidateLow = lowSeries[right];

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
        /// HTF swings on 5m data (left=10, right=3) for structural sweep levels.
        /// Matches Python _update_htf_swings().
        /// </summary>
        private void UpdateHTFSwings(double barHigh, double barLow, int bi)
        {
            int htfLeft = 10;
            int htfRight = 3;
            int ci = bi - htfRight;
            if (ci < htfLeft || ci <= 0) return;

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

        /// <summary>
        /// Find nth most recent swing (no filtering). Matches Python _find_nth_swing_price().
        /// </summary>
        private double FindNthSwing(List<SwingPoint> swings, int n, int dir,
            int entryBarIdx, double entryPrice)
        {
            int count = 0;
            for (int j = swings.Count - 1; j >= 0; j--)
            {
                count++;
                if (count == n) return swings[j].Price;
            }
            return double.NaN;
        }

        /// <summary>
        /// Find Internal Liquidity (IRL) target — nearest swing beyond entry.
        /// Matches Python logic with dynamic fallback = risk * 2.0.
        /// </summary>
        private double FindIRL(int dir, double entry, double stop)
        {
            // Match Python: use the most recent confirmed swing price (ffilled),
            // NOT "nearest swing above/below entry".
            // Python: swing_high_price[i] = shift(1).ffill() of swing highs
            // If target is on wrong side of entry, fallback to 2R.
            double risk = Math.Abs(entry - stop);
            if (dir == 1)
            {
                // Most recent swing high (equivalent to Python's swing_high_price ffilled)
                double target = double.NaN;
                if (swingHighs.Count > 0)
                    target = swingHighs[swingHighs.Count - 1].Price;

                if (double.IsNaN(target) || target <= entry)
                    return risk > 0 ? entry + risk * 2.0 : entry + 20.0;
                return target;
            }
            else
            {
                double target = double.NaN;
                if (swingLows.Count > 0)
                    target = swingLows[swingLows.Count - 1].Price;

                if (double.IsNaN(target) || target >= entry)
                    return risk > 0 ? entry - risk * 2.0 : entry - 20.0;
                return target;
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

        /// <summary>
        /// Initialize rollover dates for NQ quarterly rolls (2016-2027).
        /// 2nd Thursday of March, June, Sept, Dec +/- 1 day.
        /// </summary>
        private void InitRollDates()
        {
            int[] rollMonths = { 3, 6, 9, 12 };
            for (int year = 2016; year <= 2027; year++)
            {
                foreach (int month in rollMonths)
                {
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
                foreach (var line in lines.Skip(1))
                {
                    var parts = line.Split(',');
                    if (parts.Length < 4) continue;

                    string impact = parts[3].Trim().ToLower();
                    if (impact != "high") continue;

                    string dateStr = parts[0].Trim();
                    string timeStr = parts[1].Trim();

                    if (DateTime.TryParse(dateStr + " " + timeStr, out DateTime eventTime))
                        newsEventTimes.Add(eventTime);
                }
                Print(string.Format("Loaded {0} high-impact news events", newsEventTimes.Count));
            }
            catch (Exception ex)
            {
                Print("Failed to load news calendar: " + ex.Message);
            }
        }

        #endregion

        #region CSV Signal Loading

        private void LoadSignalCSV()
        {
            if (string.IsNullOrEmpty(SignalCSVPath))
            {
                Print("WARNING: SignalCSVPath is empty -- no CSV signals will fire.");
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
                if (lines.Length < 2) { Print("WARNING: Signal CSV has no data rows."); return; }

                string[] header = lines[0].Split(',');
                var colIdx = new Dictionary<string, int>();
                for (int i = 0; i < header.Length; i++)
                    colIdx[header[i].Trim().ToLower()] = i;

                string[] required = { "bar_time_et", "signal_dir", "signal_type", "entry_price", "model_stop" };
                foreach (var col in required)
                {
                    if (!colIdx.ContainsKey(col))
                    { Print("ERROR: Signal CSV missing column: " + col); return; }
                }

                int loaded = 0, skipped = 0;
                for (int row = 1; row < lines.Length; row++)
                {
                    string line = lines[row].Trim();
                    if (string.IsNullOrEmpty(line)) continue;
                    string[] parts = line.Split(',');
                    if (parts.Length < required.Length) { skipped++; continue; }

                    string timeStr = parts[colIdx["bar_time_et"]].Trim();
                    if (!DateTime.TryParseExact(timeStr, "yyyy-MM-dd HH:mm:ss",
                        System.Globalization.CultureInfo.InvariantCulture,
                        System.Globalization.DateTimeStyles.None, out DateTime barTimeET))
                    { skipped++; continue; }

                    if (!int.TryParse(parts[colIdx["signal_dir"]].Trim(), out int dir))
                    { skipped++; continue; }

                    if (!double.TryParse(parts[colIdx["entry_price"]].Trim(),
                        System.Globalization.NumberStyles.Any,
                        System.Globalization.CultureInfo.InvariantCulture, out double entryPrice))
                    { skipped++; continue; }

                    if (!double.TryParse(parts[colIdx["model_stop"]].Trim(),
                        System.Globalization.NumberStyles.Any,
                        System.Globalization.CultureInfo.InvariantCulture, out double modelStop))
                    { skipped++; continue; }

                    var sig = new SignalRow
                    {
                        Direction = dir,
                        Type = GetCSVStr(parts, colIdx, "signal_type", "trend"),
                        EntryPrice = entryPrice,
                        ModelStop = modelStop,
                        IRL_Target = GetCSVDbl(parts, colIdx, "irl_target", 0),
                        HasSMT = GetCSVBool(parts, colIdx, "has_smt", false),
                        Grade = GetCSVStr(parts, colIdx, "grade", "B+"),
                        BiasDirection = GetCSVInt(parts, colIdx, "bias_direction", 0),
                        Regime = GetCSVDbl(parts, colIdx, "regime", 0.5),
                        SweepScore = GetCSVInt(parts, colIdx, "sweep_score", 0)
                    };

                    long key = barTimeET.Ticks;
                    if (!signalMap.ContainsKey(key))
                    { signalMap[key] = sig; loaded++; }
                    else
                    {
                        if (sig.SweepScore > signalMap[key].SweepScore)
                            signalMap[key] = sig;
                        skipped++;
                    }
                }

                Print(string.Format("Signal CSV: loaded {0} signals, skipped {1}", loaded, skipped));
            }
            catch (Exception ex)
            {
                Print("ERROR loading signal CSV: " + ex.Message);
            }
        }

        private string GetCSVStr(string[] parts, Dictionary<string, int> idx, string col, string def)
        {
            if (!idx.ContainsKey(col) || idx[col] >= parts.Length) return def;
            string val = parts[idx[col]].Trim();
            return string.IsNullOrEmpty(val) ? def : val;
        }

        private double GetCSVDbl(string[] parts, Dictionary<string, int> idx, string col, double def)
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
    }
}
