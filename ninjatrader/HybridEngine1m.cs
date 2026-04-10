// =============================================================================
// HybridEngine1m.cs — NinjaTrader 8 NinjaScript
// 1m Hybrid Engine: 5m Zone Detection + 1m Execution
//
// EXACT port of experiments/unified_engine_1m.py + all feature modules:
//   features/bias.py, features/sessions.py, features/displacement.py,
//   features/fvg.py, features/news_filter.py
//
// Data series:
//   BIP 0: NQ/MNQ 1min  (execution)
//   BIP 1: NQ/MNQ 5min  (zone detection, swing, ATR, FVG)
//   BIP 2: NQ/MNQ 60min (1H HTF bias — FVG tracking)
//   BIP 3: NQ/MNQ 240min(4H HTF bias — FVG tracking + fluency)
//
// Worst-case: 1550t, PF=2.32, R=+414.4, PPDD=43.0, MaxDD=9.6R
// =============================================================================

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Linq;
using System.Text;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class HybridEngine1m : Strategy
    {
        // ================================================================
        // PARAMETERS — all from config/params.yaml
        // ================================================================

        #region 1. FVG Zone
        [NinjaScriptProperty][Display(Name="FVG Size ATR Mult",Order=1,GroupName="01 FVG")]
        public double FvgSizeAtrMult { get; set; }
        [NinjaScriptProperty][Display(Name="Max FVG Age (5m bars)",Order=2,GroupName="01 FVG")]
        public int MaxFvgAge { get; set; }
        [NinjaScriptProperty][Display(Name="Max Wait (bd→FVG)",Order=3,GroupName="01 FVG")]
        public int MaxWaitBars { get; set; }
        [NinjaScriptProperty][Display(Name="Min Breakdown Depth Pts",Order=4,GroupName="01 FVG")]
        public double MinBreakdownDepth { get; set; }
        #endregion

        #region 2. Stop
        [NinjaScriptProperty][Display(Name="Stop Buffer Pct",Order=1,GroupName="02 Stop")]
        public double StopBufferPct { get; set; }
        [NinjaScriptProperty][Display(Name="Tighten Factor",Order=2,GroupName="02 Stop")]
        public double TightenFactor { get; set; }
        [NinjaScriptProperty][Display(Name="Min Stop ATR Mult",Order=3,GroupName="02 Stop")]
        public double MinStopAtrMult { get; set; }
        #endregion

        #region 3. Exit
        [NinjaScriptProperty][Display(Name="Trim Pct",Order=1,GroupName="03 Exit")]
        public double TrimPct { get; set; }
        [NinjaScriptProperty][Display(Name="TP1 R Multiple",Order=2,GroupName="03 Exit")]
        public double FixedTpR { get; set; }
        [NinjaScriptProperty][Display(Name="Nth Swing Trail",Order=3,GroupName="03 Exit")]
        public int NthSwing { get; set; }
        [NinjaScriptProperty][Display(Name="Use BE",Order=4,GroupName="03 Exit")]
        public bool UseBE { get; set; }
        [NinjaScriptProperty][Display(Name="BE Offset R",Order=5,GroupName="03 Exit")]
        public double BeOffsetR { get; set; }
        [NinjaScriptProperty][Display(Name="Worst Case Trim BE",Order=6,GroupName="03 Exit")]
        public bool WorstCaseTrimBe { get; set; }
        #endregion

        #region 4. Risk
        [NinjaScriptProperty][Display(Name="Normal R ($)",Order=1,GroupName="04 Risk")]
        public double NormalR { get; set; }
        [NinjaScriptProperty][Display(Name="Reduced R ($)",Order=2,GroupName="04 Risk")]
        public double ReducedR { get; set; }
        [NinjaScriptProperty][Display(Name="Daily Max Loss R",Order=3,GroupName="04 Risk")]
        public double DailyMaxLossR { get; set; }
        [NinjaScriptProperty][Display(Name="Max Consec Losses",Order=4,GroupName="04 Risk")]
        public int MaxConsecLosses { get; set; }
        [NinjaScriptProperty][Display(Name="Max Positions",Order=5,GroupName="04 Risk")]
        public int MaxPositions { get; set; }
        #endregion

        #region 5. Sizing
        [NinjaScriptProperty][Display(Name="A+ Mult",Order=1,GroupName="05 Size")]
        public double APlusMult { get; set; }
        [NinjaScriptProperty][Display(Name="B+ Mult",Order=2,GroupName="05 Size")]
        public double BPlusMult { get; set; }
        [NinjaScriptProperty][Display(Name="Big Sweep Threshold",Order=3,GroupName="05 Size")]
        public double BigSweepThreshold { get; set; }
        [NinjaScriptProperty][Display(Name="Big Sweep Mult",Order=4,GroupName="05 Size")]
        public double BigSweepMult { get; set; }
        [NinjaScriptProperty][Display(Name="AM Short Mult",Order=5,GroupName="05 Size")]
        public double AMShortMult { get; set; }
        [NinjaScriptProperty][Display(Name="Trend R Mult",Order=6,GroupName="05 Size")]
        public double TrendRMult { get; set; }
        #endregion

        #region 6. Execution
        [NinjaScriptProperty][Display(Name="Slippage Ticks",Order=1,GroupName="06 Exec")]
        public int SlippageTicks { get; set; }
        [NinjaScriptProperty][Display(Name="Comm Per Side",Order=2,GroupName="06 Exec")]
        public double CommPerSide { get; set; }
        [NinjaScriptProperty][Display(Name="EOD Close",Order=3,GroupName="06 Exec")]
        public bool EODClose { get; set; }
        #endregion

        #region 7. Swing
        [NinjaScriptProperty][Display(Name="Left Bars",Order=1,GroupName="07 Swing")]
        public int SwingLeftBars { get; set; }
        [NinjaScriptProperty][Display(Name="Right Bars",Order=2,GroupName="07 Swing")]
        public int SwingRightBars { get; set; }
        #endregion

        #region 8. Tier
        [NinjaScriptProperty][Display(Name="Enable Chain",Order=1,GroupName="08 Tier")]
        public bool EnableChain { get; set; }
        [NinjaScriptProperty][Display(Name="Enable Trend",Order=2,GroupName="08 Tier")]
        public bool EnableTrend { get; set; }
        [NinjaScriptProperty][Display(Name="PD Threshold",Order=3,GroupName="08 Tier")]
        public double PdThreshold { get; set; }
        [NinjaScriptProperty][Display(Name="BD Exclusion Bars",Order=4,GroupName="08 Tier")]
        public int BdExclusionBars { get; set; }
        #endregion

        #region 9. Bias
        [NinjaScriptProperty][Display(Name="Bias HTF Weight",Order=1,GroupName="09 Bias")]
        public double BiasWtHtf { get; set; }
        [NinjaScriptProperty][Display(Name="Bias ON Weight",Order=2,GroupName="09 Bias")]
        public double BiasWtOn { get; set; }
        [NinjaScriptProperty][Display(Name="Bias ORM Weight",Order=3,GroupName="09 Bias")]
        public double BiasWtOrm { get; set; }
        [NinjaScriptProperty][Display(Name="Bias Threshold",Order=4,GroupName="09 Bias")]
        public double BiasThreshold { get; set; }
        [NinjaScriptProperty][Display(Name="ON Bull Threshold",Order=5,GroupName="09 Bias")]
        public double OnBullThresh { get; set; }
        [NinjaScriptProperty][Display(Name="ON Bear Threshold",Order=6,GroupName="09 Bias")]
        public double OnBearThresh { get; set; }
        [NinjaScriptProperty][Display(Name="4H Bias Weight",Order=7,GroupName="09 Bias")]
        public double Bias4hWeight { get; set; }
        [NinjaScriptProperty][Display(Name="1H Bias Weight",Order=8,GroupName="09 Bias")]
        public double Bias1hWeight { get; set; }
        #endregion

        #region 10. Fluency / Regime
        [NinjaScriptProperty][Display(Name="Fluency Window",Order=1,GroupName="10 Regime")]
        public int FluencyWindow { get; set; }
        [NinjaScriptProperty][Display(Name="Fluency Threshold",Order=2,GroupName="10 Regime")]
        public double FluencyThreshold { get; set; }
        [NinjaScriptProperty][Display(Name="Flu W Dir",Order=3,GroupName="10 Regime")]
        public double FluWDir { get; set; }
        [NinjaScriptProperty][Display(Name="Flu W Body",Order=4,GroupName="10 Regime")]
        public double FluWBody { get; set; }
        [NinjaScriptProperty][Display(Name="Flu W Size",Order=5,GroupName="10 Regime")]
        public double FluWSize { get; set; }
        [NinjaScriptProperty][Display(Name="Chop Range Points",Order=6,GroupName="10 Regime")]
        public double ChopRangePoints { get; set; }
        [NinjaScriptProperty][Display(Name="Chop Window Bars (5m)",Order=7,GroupName="10 Regime")]
        public int ChopWindowBars { get; set; }
        #endregion

        #region 11. News
        [NinjaScriptProperty][Display(Name="News CSV Path",Order=1,GroupName="11 News")]
        public string NewsCsvPath { get; set; }
        [NinjaScriptProperty][Display(Name="Blackout Min Before",Order=2,GroupName="11 News")]
        public int NewsBlackoutBefore { get; set; }
        [NinjaScriptProperty][Display(Name="Cooldown Min After",Order=3,GroupName="11 News")]
        public int NewsCooldownAfter { get; set; }
        #endregion

        // ================================================================
        // DATA CLASSES
        // ================================================================

        #region Data Classes

        private class UnifiedZone
        {
            public int Dir;               // +1 bull, -1 bear
            public double Top, Bottom, Size;
            public int BirthBar5m;
            public double BirthAtr;
            public int Tier;              // 1=chain, 2=trend
            public double SweepRangeAtr;
            public bool Used;
        }

        private class Pos
        {
            public int Tier, Dir;
            public int EntryBar1m;
            public double Entry, Stop, TP1;
            public int Contracts, Orig, Remaining;
            public bool Trimmed;
            public double BeStop, TrailStop;
            public string Grade;
            public double SweepAtr;
        }

        private class Breakdown
        {
            public int Bar5m;
            public string LevelType; // "low" or "high"
        }

        private class HtfFvg
        {
            public int Dir;           // +1 bull, -1 bear
            public double Top, Bottom;
            public string Status;     // untested, tested_rejected, invalidated
            public HtfFvg() { Status = "untested"; }
        }

        private class NewsEvent
        {
            public DateTime StartUtc; // blackout start
            public DateTime EndUtc;   // blackout end
        }

        #endregion

        // ================================================================
        // INTERNAL STATE
        // ================================================================

        // Indicators
        private ATR atr5m;
        private ATR atr4h;

        // Geometry
        private double pointValue, slipPts;

        // Bar counters
        private int bar5m, bar1h, bar4h;

        // --- Zones ---
        private List<UnifiedZone> activeZones;
        private HashSet<int> bdTerritory;
        private List<Breakdown> recentBds;

        // --- Positions ---
        private List<Pos> positions;

        // --- 5m Swing (with bar index for NOT-MSS search) ---
        private struct SwingPt { public int Bar; public double Price; }
        private List<SwingPt> swHiPts, swLoPts;

        // --- Session levels (computed from 5m) ---
        // Overnight (18:00-09:30 ET)
        private double onHigh, onLow;         // accumulating
        private double onHighReady, onLowReady; // completed (usable)
        private bool onReady;
        private DateTime onSessionDate;
        // NY open
        private double nyOpen;
        private bool nyOpenSet;
        // ORM (09:30-10:00 ET)
        private double ormHigh, ormLow;
        private double ormHighReady, ormLowReady;
        private bool ormReady;

        // --- HTF FVGs (1H + 4H) ---
        private List<HtfFvg> fvgs1h, fvgs4h;
        // Pending FVG detection (shift(1))
        private bool pend1hBull, pend1hBear;
        private double p1hBullT, p1hBullB, p1hBearT, p1hBearB;
        private bool pend4hBull, pend4hBear;
        private double p4hBullT, p4hBullB, p4hBearT, p4hBearB;
        // HTF bias (shifted: only updates on TF bar close, then held)
        private double htfBias;    // composite HTF bias [-1,+1]
        private int htfPdaCount;   // total active FVGs on 1H+4H

        // --- 4H Fluency ---
        // Ring buffer for 4H bars (for rolling fluency)
        private double[] flu4hOpen, flu4hHigh, flu4hLow, flu4hClose, flu4hAtr;
        private int flu4hIdx, flu4hCount;
        private double fluency4h;

        // --- Composite bias (recomputed each 1m bar from components) ---
        private double biasDirection; // +1, -1, or 0
        private double regimeValue;   // 1.0, 0.5, or 0.0

        // --- News ---
        private List<NewsEvent> newsEvents;

        // --- Daily state ---
        private DateTime curDate;
        private double dayPnl;
        private int consecLoss;
        private bool dayStopped;

        // --- 5m FVG detection (shift(1)) ---
        private bool pend5mBull, pend5mBear;
        private double p5mBullT, p5mBullB, p5mBearT, p5mBearB;
        private int p5mBullBar, p5mBearBar;

        // Previous close for breakdown detection
        private double prevClose5m;

        // --- Trade log ---
        private StringBuilder tradeLog;
        private int tradeCount;
        private double totalR, totWins, totLosses;

        // ================================================================
        // INITIALIZATION
        // ================================================================
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "1m Hybrid Engine — exact port of unified_engine_1m.py";
                Name = "HybridEngine1m";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = false;
                MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                StartBehavior = StartBehavior.WaitUntilFlat;
                BarsRequiredToTrade = 50;

                // params.yaml defaults + V4 (Axiom 9 structurally resolved, 2026-04-10)
                // V4: no BE + trim_pct=0.0 (trims 1 contract) + pure 5th swing trail
                // Python validated: +891R / PF=3.53 / PPDD=63.3 / 0 neg years
                FvgSizeAtrMult = 0.3;  MaxFvgAge = 200;  MaxWaitBars = 30;
                MinBreakdownDepth = 1.0;
                StopBufferPct = 0.15;  TightenFactor = 0.85;  MinStopAtrMult = 0.15;
                TrimPct = 0.0;  FixedTpR = 1.0;  NthSwing = 5;  // V4 baseline
                UseBE = false;  BeOffsetR = 0.0;  WorstCaseTrimBe = false; // V4: no BE
                NormalR = 1000;  ReducedR = 500;
                DailyMaxLossR = 2.0;  MaxConsecLosses = 2;  MaxPositions = 2;
                APlusMult = 1.5;  BPlusMult = 1.0;
                BigSweepThreshold = 1.3;  BigSweepMult = 1.5;
                AMShortMult = 0.5;  TrendRMult = 0.5;
                SlippageTicks = 1;  CommPerSide = 0.62;  EODClose = true;
                SwingLeftBars = 3;  SwingRightBars = 1;
                EnableChain = true;  EnableTrend = true;
                PdThreshold = 0.5;  BdExclusionBars = 30;

                // Bias weights (features/bias.py)
                BiasWtHtf = 0.4;  BiasWtOn = 0.3;  BiasWtOrm = 0.3;
                BiasThreshold = 0.2;
                OnBullThresh = 0.6;  OnBearThresh = 0.4;
                Bias4hWeight = 0.6;  Bias1hWeight = 0.4;

                // Fluency (features/displacement.py)
                FluencyWindow = 6;  FluencyThreshold = 0.60;
                FluWDir = 0.4;  FluWBody = 0.3;  FluWSize = 0.3;
                ChopRangePoints = 25;  ChopWindowBars = 50;

                // News
                NewsCsvPath = "";
                NewsBlackoutBefore = 60;  NewsCooldownAfter = 5;
            }
            else if (State == State.Configure)
            {
                AddDataSeries(BarsPeriodType.Minute, 5);    // BIP 1: 5m
                AddDataSeries(BarsPeriodType.Minute, 60);   // BIP 2: 1H
                AddDataSeries(BarsPeriodType.Minute, 240);  // BIP 3: 4H
            }
            else if (State == State.DataLoaded)
            {
                atr5m = ATR(BarsArray[1], 14);
                atr4h = ATR(BarsArray[3], 14);
                pointValue = Instrument.MasterInstrument.PointValue;
                slipPts = SlippageTicks * TickSize;

                activeZones = new List<UnifiedZone>();
                bdTerritory = new HashSet<int>();
                recentBds = new List<Breakdown>();
                positions = new List<Pos>();
                swHiPts = new List<SwingPt>();
                swLoPts = new List<SwingPt>();
                fvgs1h = new List<HtfFvg>();
                fvgs4h = new List<HtfFvg>();
                newsEvents = new List<NewsEvent>();

                // 4H fluency ring buffer (window size)
                int bufSz = Math.Max(FluencyWindow, 30);
                flu4hOpen = new double[bufSz];
                flu4hHigh = new double[bufSz];
                flu4hLow = new double[bufSz];
                flu4hClose = new double[bufSz];
                flu4hAtr = new double[bufSz];
                flu4hIdx = 0; flu4hCount = 0;
                fluency4h = 0;

                onHigh = double.NaN; onLow = double.NaN;
                onHighReady = double.NaN; onLowReady = double.NaN;
                onReady = false; onSessionDate = DateTime.MinValue;
                nyOpen = double.NaN; nyOpenSet = false;
                ormHigh = double.NaN; ormLow = double.NaN;
                ormHighReady = double.NaN; ormLowReady = double.NaN;
                ormReady = false;

                htfBias = 0; htfPdaCount = 0;
                biasDirection = 0; regimeValue = 0.5;

                curDate = DateTime.MinValue;
                dayPnl = 0; consecLoss = 0; dayStopped = false;
                prevClose5m = double.NaN;
                bar5m = 0; bar1h = 0; bar4h = 0;

                tradeLog = new StringBuilder();
                tradeLog.AppendLine("entry_time,exit_time,dir,tier,grade,entry_price,exit_price," +
                    "stop_price,tp1_price,reason,r,pnl_dollars,contracts,stop_dist,trimmed");
                tradeCount = 0; totalR = 0; totWins = 0; totLosses = 0;

                LoadNewsCalendar();
            }
            else if (State == State.Terminated)
            {
                if (tradeLog != null && tradeCount > 0)
                {
                    double pf = totLosses > 0 ? totWins / totLosses : 999;
                    string path = Path.Combine(
                        Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                        "Hybrid1m_NT8_trades.csv");
                    File.WriteAllText(path, tradeLog.ToString());
                    Print($"HYBRID 1m: {tradeCount}t R={totalR:+0.0} PF={pf:F2} CSV={path}");
                }
            }
        }

        // ================================================================
        // MAIN ROUTER
        // ================================================================
        protected override void OnBarUpdate()
        {
            switch (BarsInProgress)
            {
                case 0: OnBar1m(); break;
                case 1: OnBar5m(); break;
                case 2: OnBar1h(); break;
                case 3: OnBar4h(); break;
            }
        }

        // ================================================================
        // 5m BAR: Zone detection, breakdown, swing, session levels
        // ================================================================
        private void OnBar5m()
        {
            if (CurrentBars[1] < 50) return;
            bar5m++;

            double curAtr = atr5m[0];
            if (double.IsNaN(curAtr) || curAtr <= 0) curAtr = 30.0;

            DateTime bc = Times[1][0];
            DateTime bo = bc.AddMinutes(-5);
            double hf = bo.Hour + bo.Minute / 60.0;

            UpdateSessionLevels(bo, hf);
            EmitPending5mFVGs(curAtr, hf);
            Detect5mFVGs();
            DetectBreakdowns5m();
            UpdateSwings5m();
            // Zone invalidation moved to OnBar1m only (matches Python)

            prevClose5m = Closes[1][0];
        }

        // ================================================================
        // 1H BAR: HTF FVG detection for bias
        // ================================================================
        private void OnBar1h()
        {
            if (CurrentBars[2] < 3) return;
            bar1h++;

            // Emit pending 1H FVGs (shift by 1)
            if (pend1hBull)
            {
                fvgs1h.Add(new HtfFvg { Dir = 1, Top = p1hBullT, Bottom = p1hBullB });
                pend1hBull = false;
            }
            if (pend1hBear)
            {
                fvgs1h.Add(new HtfFvg { Dir = -1, Top = p1hBearT, Bottom = p1hBearB });
                pend1hBear = false;
            }

            // Detect new 1H FVGs
            if (CurrentBars[2] >= 3)
            {
                double h1 = Highs[2][2], l1 = Lows[2][2];
                double h3 = Highs[2][0], l3 = Lows[2][0];
                if (h1 < l3) { pend1hBull = true; p1hBullT = l3; p1hBullB = h1; }
                if (l1 > h3) { pend1hBear = true; p1hBearT = l1; p1hBearB = h3; }
            }

            // Update 1H FVG states
            UpdateHtfFvgStates(fvgs1h, Highs[2][0], Lows[2][0], Closes[2][0]);

            // Recompute HTF bias
            RecomputeHtfBias(Closes[2][0]);
        }

        // ================================================================
        // 4H BAR: HTF FVG detection + fluency
        // ================================================================
        private void OnBar4h()
        {
            if (CurrentBars[3] < 3) return;
            bar4h++;

            // Emit pending 4H FVGs (shift by 1)
            if (pend4hBull)
            {
                fvgs4h.Add(new HtfFvg { Dir = 1, Top = p4hBullT, Bottom = p4hBullB });
                pend4hBull = false;
            }
            if (pend4hBear)
            {
                fvgs4h.Add(new HtfFvg { Dir = -1, Top = p4hBearT, Bottom = p4hBearB });
                pend4hBear = false;
            }

            // Detect new 4H FVGs
            if (CurrentBars[3] >= 3)
            {
                double h1 = Highs[3][2], l1 = Lows[3][2];
                double h3 = Highs[3][0], l3 = Lows[3][0];
                if (h1 < l3) { pend4hBull = true; p4hBullT = l3; p4hBullB = h1; }
                if (l1 > h3) { pend4hBear = true; p4hBearT = l1; p4hBearB = h3; }
            }

            // Update 4H FVG states
            UpdateHtfFvgStates(fvgs4h, Highs[3][0], Lows[3][0], Closes[3][0]);

            // Compute 4H fluency
            Compute4hFluency();

            // Recompute HTF bias
            RecomputeHtfBias(Closes[3][0]);
        }

        // ================================================================
        // HTF FVG STATE MACHINE (shared 1H/4H)
        // Python: features/fvg.py _update_fvg_status
        // ================================================================
        private void UpdateHtfFvgStates(List<HtfFvg> fvgs, double barH, double barL, double barC)
        {
            for (int i = fvgs.Count - 1; i >= 0; i--)
            {
                var f = fvgs[i];
                if (f.Status == "invalidated") { fvgs.RemoveAt(i); continue; }

                if (f.Dir == 1) // Bull FVG (support below price)
                {
                    bool entered = barL <= f.Top && barH >= f.Bottom;
                    if (entered)
                    {
                        if (barC < f.Bottom) f.Status = "invalidated";
                        else f.Status = "tested_rejected";
                    }
                    else if (barC < f.Bottom) f.Status = "invalidated";
                }
                else // Bear FVG (resistance above price)
                {
                    bool entered = barH >= f.Bottom && barL <= f.Top;
                    if (entered)
                    {
                        if (barC > f.Top) f.Status = "invalidated";
                        else f.Status = "tested_rejected";
                    }
                    else if (barC > f.Top) f.Status = "invalidated";
                }

                if (f.Status == "invalidated") fvgs.RemoveAt(i);
            }
        }

        // ================================================================
        // HTF BIAS — Python: features/bias.py compute_htf_bias
        // composite = 0.6 * bias_4h + 0.4 * bias_1h
        // ================================================================
        private void RecomputeHtfBias(double curPrice)
        {
            double bias4h = ComputeDrawBias(fvgs4h, curPrice);
            double bias1h = ComputeDrawBias(fvgs1h, curPrice);

            // Fluency dampening: if 4H fluency < threshold, halve
            double fluFactor = fluency4h >= FluencyThreshold ? 1.0 : 0.5;
            htfBias = (Bias4hWeight * bias4h + Bias1hWeight * bias1h) * fluFactor;
            htfBias = Math.Max(-1.0, Math.Min(1.0, htfBias));

            // PDA count = total active FVGs
            htfPdaCount = fvgs4h.Count + fvgs1h.Count;
        }

        private double ComputeDrawBias(List<HtfFvg> fvgs, double price)
        {
            // Count active FVGs by direction relative to price
            int bullAbove = 0, bearBelow = 0;
            foreach (var f in fvgs)
            {
                double mid = (f.Top + f.Bottom) / 2;
                if (f.Dir == 1 && mid > price) bullAbove++;
                if (f.Dir == -1 && mid < price) bearBelow++;
            }
            if (bullAbove > 0 && bearBelow == 0) return 1.0;
            if (bearBelow > 0 && bullAbove == 0) return -1.0;
            if (bullAbove > 0 && bearBelow > 0)
                return Math.Sign(bullAbove - bearBelow) * 0.5;
            return 0.0;
        }

        // ================================================================
        // 4H FLUENCY — Python: features/displacement.py compute_fluency
        // ================================================================
        private void Compute4hFluency()
        {
            // Push current 4H bar into ring buffer
            int sz = flu4hOpen.Length;
            flu4hOpen[flu4hIdx % sz] = Opens[3][0];
            flu4hHigh[flu4hIdx % sz] = Highs[3][0];
            flu4hLow[flu4hIdx % sz] = Lows[3][0];
            flu4hClose[flu4hIdx % sz] = Closes[3][0];
            double a4h = atr4h[0];
            flu4hAtr[flu4hIdx % sz] = (double.IsNaN(a4h) || a4h <= 0) ? 30.0 : a4h;
            flu4hIdx++;
            flu4hCount = Math.Min(flu4hCount + 1, sz);

            if (flu4hCount < FluencyWindow) { fluency4h = 0; return; }

            int w = FluencyWindow;
            int bullCount = 0, n = 0;
            double bodyRatioSum = 0, barSizeSum = 0;

            for (int j = 0; j < w; j++)
            {
                int idx = ((flu4hIdx - 1 - j) % sz + sz) % sz;
                double o = flu4hOpen[idx], h = flu4hHigh[idx];
                double l = flu4hLow[idx], c = flu4hClose[idx];
                double atr = flu4hAtr[idx];

                double body = Math.Abs(c - o);
                double range = h - l;
                if (range <= 0) continue;

                if (c > o) bullCount++;
                bodyRatioSum += body / range;
                double barSzNorm = atr > 0 ? Math.Min(range / atr, 2.0) : 0;
                barSizeSum += barSzNorm;
                n++;
            }
            if (n < 2) { fluency4h = 0; return; }

            double dirRatio = (double)Math.Max(bullCount, n - bullCount) / n;
            double avgBodyRatio = bodyRatioSum / n;
            double avgBarSize = Math.Min(barSizeSum / n, 1.0);

            fluency4h = FluWDir * dirRatio + FluWBody * avgBodyRatio + FluWSize * avgBarSize;
            fluency4h = Math.Max(0, Math.Min(1, fluency4h));
        }

        // ================================================================
        // SESSION LEVELS — Python: features/sessions.py
        // Overnight (18:00-09:30), NY open (09:30), ORM (09:30-10:00)
        // ================================================================
        private void UpdateSessionLevels(DateTime barOpenET, double hf)
        {
            DateTime sessDate = barOpenET.Hour >= 18
                ? barOpenET.Date.AddDays(1)
                : barOpenET.Date;

            // New session day
            if (sessDate != onSessionDate)
            {
                // Finalize previous ON levels
                if (!double.IsNaN(onHigh))
                {
                    onHighReady = onHigh;
                    onLowReady = onLow;
                    onReady = true;
                }
                // Finalize previous ORM
                if (!double.IsNaN(ormHigh))
                {
                    ormHighReady = ormHigh;
                    ormLowReady = ormLow;
                    ormReady = true;
                }
                onSessionDate = sessDate;
                onHigh = double.NaN; onLow = double.NaN;
                nyOpen = double.NaN; nyOpenSet = false;
                ormHigh = double.NaN; ormLow = double.NaN;
                ormReady = false;  // FIX: reset so new day's ORM gets finalized
            }

            double h = Highs[1][0], l = Lows[1][0];

            // Overnight accumulation (18:00 - 09:30)
            if (hf >= 18.0 || hf < 9.5)
            {
                if (double.IsNaN(onHigh) || h > onHigh) onHigh = h;
                if (double.IsNaN(onLow) || l < onLow) onLow = l;
            }

            // NY open (first 5m bar at 09:30)
            if (!nyOpenSet && hf >= 9.5 && hf < 9.6)
            {
                nyOpen = Opens[1][0];
                nyOpenSet = true;
                // Finalize ON at NY open
                if (!double.IsNaN(onHigh))
                {
                    onHighReady = onHigh;
                    onLowReady = onLow;
                    onReady = true;
                }
            }

            // ORM accumulation (09:30 - 10:00)
            if (hf >= 9.5 && hf < 10.0)
            {
                if (double.IsNaN(ormHigh) || h > ormHigh) ormHigh = h;
                if (double.IsNaN(ormLow) || l < ormLow) ormLow = l;
            }
            // Finalize ORM at 10:00
            else if (hf >= 10.0 && !double.IsNaN(ormHigh) && !ormReady)
            {
                ormHighReady = ormHigh;
                ormLowReady = ormLow;
                ormReady = true;
            }
        }

        // ================================================================
        // COMPOSITE BIAS — Python: features/bias.py get_composite_bias
        // composite = 0.4*HTF + 0.3*ON + 0.3*ORM
        // ================================================================
        private void RecomputeCompositeBias()
        {
            // 1. HTF component (already computed in htfBias)
            double bHtf = htfBias;

            // 2. Overnight component
            double bOn = 0;
            if (onReady && !double.IsNaN(onHighReady) && !double.IsNaN(onLowReady)
                && !double.IsNaN(nyOpen))
            {
                double onRange = onHighReady - onLowReady;
                if (onRange > 0)
                {
                    double pos = (nyOpen - onLowReady) / onRange;
                    if (pos > OnBullThresh) bOn = 1.0;
                    else if (pos < OnBearThresh) bOn = -1.0;
                }
            }

            // 3. ORM component
            double bOrm = 0;
            if (ormReady && onReady
                && !double.IsNaN(ormHighReady) && !double.IsNaN(ormLowReady)
                && !double.IsNaN(onHighReady) && !double.IsNaN(onLowReady))
            {
                bool brokeAbove = ormHighReady > onHighReady;
                bool brokeBelow = ormLowReady < onLowReady;
                if (brokeAbove && !brokeBelow) bOrm = 1.0;
                else if (brokeBelow && !brokeAbove) bOrm = -1.0;
            }

            // Composite
            double composite = BiasWtHtf * bHtf + BiasWtOn * bOn + BiasWtOrm * bOrm;
            composite = Math.Max(-1.0, Math.Min(1.0, composite));

            // Discretize: |composite| > 0.2 → direction
            if (composite > BiasThreshold) biasDirection = 1.0;
            else if (composite < -BiasThreshold) biasDirection = -1.0;
            else biasDirection = 0.0;

            // Regime: PDA > 0 AND fluent AND not choppy → 1.0
            if (htfPdaCount <= 0) regimeValue = 0.0;
            else if (fluency4h >= FluencyThreshold && !IsChoppy()) regimeValue = 1.0;
            else regimeValue = 0.5;
        }

        private bool IsChoppy()
        {
            // Check rolling range on 5m bars
            if (CurrentBars[1] < ChopWindowBars) return false;
            double maxH = double.MinValue, minL = double.MaxValue;
            for (int j = 0; j < ChopWindowBars; j++)
            {
                if (Highs[1][j] > maxH) maxH = Highs[1][j];
                if (Lows[1][j] < minL) minL = Lows[1][j];
            }
            return (maxH - minL) < ChopRangePoints;
        }

        // ================================================================
        // 5m FVG DETECTION (shift(1) anti-lookahead)
        // ================================================================
        private void Detect5mFVGs()
        {
            if (CurrentBars[1] < 3) return;
            double h1 = Highs[1][2], l1 = Lows[1][2];
            double h3 = Highs[1][0], l3 = Lows[1][0];
            if (h1 < l3) { pend5mBull = true; p5mBullT = l3; p5mBullB = h1; p5mBullBar = bar5m; }
            if (l1 > h3) { pend5mBear = true; p5mBearT = l1; p5mBearB = h3; p5mBearBar = bar5m; }
        }

        private void EmitPending5mFVGs(double curAtr, double hf)
        {
            if (pend5mBull)
            {
                double sz = p5mBullT - p5mBullB;
                if (sz >= FvgSizeAtrMult * curAtr)
                    RegisterZone(1, p5mBullT, p5mBullB, sz, p5mBullBar, curAtr);
                pend5mBull = false;
            }
            if (pend5mBear)
            {
                double sz = p5mBearT - p5mBearB;
                if (sz >= FvgSizeAtrMult * curAtr)
                    RegisterZone(-1, p5mBearT, p5mBearB, sz, p5mBearBar, curAtr);
                pend5mBear = false;
            }
        }

        // ================================================================
        // ZONE REGISTRATION (chain vs trend classification)
        // ================================================================
        private void RegisterZone(int dir, double top, double bottom, double size,
                                   int birthBar, double atr)
        {
            // --- Chain: breakdown → NOT-MSS FVG ---
            if (EnableChain)
            {
                foreach (var bd in recentBds)
                {
                    if (birthBar <= bd.Bar5m || birthBar - bd.Bar5m > MaxWaitBars) continue;
                    int expectedDir = bd.LevelType == "low" ? 1 : -1;
                    if (dir != expectedDir) continue;
                    if (!PassesNotMSS(dir, birthBar)) continue;

                    double sweepAtr = 0;
                    int ago = bar5m - bd.Bar5m;
                    if (ago >= 0 && ago < CurrentBars[1])
                    {
                        double rng = Highs[1][ago] - Lows[1][ago];
                        sweepAtr = atr > 0 ? rng / atr : 0;
                    }
                    activeZones.Add(new UnifiedZone {
                        Dir = dir, Top = top, Bottom = bottom, Size = size,
                        BirthBar5m = birthBar, BirthAtr = atr, Tier = 1,
                        SweepRangeAtr = sweepAtr, Used = false,
                    });
                    return;
                }
            }

            // --- Trend: PD + bias + not in breakdown territory ---
            if (!EnableTrend || bdTerritory.Contains(birthBar)) return;
            if (!onReady || double.IsNaN(onHighReady) || double.IsNaN(onLowReady)) return;

            double onRange = onHighReady - onLowReady;
            if (onRange <= 0) return;
            double mid = (top + bottom) / 2;
            double pdPos = (mid - onLowReady) / onRange;
            if (dir == 1 && pdPos >= PdThreshold) return;
            if (dir == -1 && pdPos < PdThreshold) return;

            // Bias alignment
            if (dir == 1 && biasDirection < 0) return;
            if (dir == -1 && biasDirection > 0) return;

            activeZones.Add(new UnifiedZone {
                Dir = dir, Top = top, Bottom = bottom, Size = size,
                BirthBar5m = birthBar, BirthAtr = atr, Tier = 2,
                SweepRangeAtr = 0, Used = false,
            });
        }

        // ================================================================
        // NOT-MSS CHECK — Python: chain_engine.py find_fvg_not_mss
        // ================================================================
        private bool PassesNotMSS(int dir, int fvgBar)
        {
            // Displacement bar = 1 bar before FVG bar
            int dispBar = fvgBar - 1;
            int dispAgo = bar5m - fvgBar + 1;
            if (dispAgo < 0 || dispAgo >= CurrentBars[1]) return false;

            // Python: search backward from disp_bar up to 100 bars for first swing
            if (dir == 1) // bull FVG: check if displacement candle breaks a swing high
            {
                double dispH = Highs[1][dispAgo];
                for (int i = swHiPts.Count - 1; i >= 0; i--)
                {
                    if (swHiPts[i].Bar > dispBar) continue; // swing after disp bar, skip
                    if (dispBar - swHiPts[i].Bar > 100) break; // beyond 100-bar search range
                    // Found nearest swing high before disp bar
                    if (dispH > swHiPts[i].Price)
                        return false; // breaks swing = MSS, reject
                    break; // only check first (nearest) swing found
                }
            }
            else // bear FVG: check if displacement candle breaks a swing low
            {
                double dispL = Lows[1][dispAgo];
                for (int i = swLoPts.Count - 1; i >= 0; i--)
                {
                    if (swLoPts[i].Bar > dispBar) continue;
                    if (dispBar - swLoPts[i].Bar > 100) break;
                    if (dispL < swLoPts[i].Price)
                        return false; // breaks swing = MSS, reject
                    break;
                }
            }
            return true; // NOT MSS — passes
        }

        // ================================================================
        // BREAKDOWN DETECTION — Python: chain_engine.py detect_breakdowns
        // ================================================================
        private void DetectBreakdowns5m()
        {
            if (bar5m < 2 || !onReady || double.IsNaN(prevClose5m)) return;
            double h = Highs[1][0], l = Lows[1][0], c = Closes[1][0];

            bool tooClose = recentBds.Count > 0 && bar5m - recentBds[recentBds.Count - 1].Bar5m < 3;

            // Low breakdown: close < ON low (sweep → bull reversal)
            if (!tooClose && !double.IsNaN(onLowReady) && onLowReady > 0
                && c < onLowReady - MinBreakdownDepth && l < onLowReady && prevClose5m >= onLowReady)
            {
                recentBds.Add(new Breakdown { Bar5m = bar5m, LevelType = "low" });
                MarkBdTerritory(bar5m);
                tooClose = true; // prevent double trigger
            }
            // High breakdown: close > ON high (sweep → bear reversal)
            if (!tooClose && !double.IsNaN(onHighReady) && onHighReady > 0
                && c > onHighReady + MinBreakdownDepth && h > onHighReady && prevClose5m <= onHighReady)
            {
                recentBds.Add(new Breakdown { Bar5m = bar5m, LevelType = "high" });
                MarkBdTerritory(bar5m);
            }

            recentBds.RemoveAll(b => bar5m - b.Bar5m > MaxWaitBars * 2);
        }

        private void MarkBdTerritory(int b)
        {
            for (int j = b; j <= b + MaxWaitBars; j++)
                bdTerritory.Add(j);
        }

        // ================================================================
        // SWING DETECTION (5m fractal)
        // ================================================================
        private void UpdateSwings5m()
        {
            int need = SwingLeftBars + SwingRightBars + 1;
            if (CurrentBars[1] < need) return;
            int ck = SwingRightBars;

            bool isHi = true;
            double ch = Highs[1][ck];
            for (int j = 1; j <= SwingLeftBars && isHi; j++)
                if (Highs[1][ck + j] >= ch) isHi = false;
            for (int j = 1; j <= SwingRightBars && isHi; j++)
                if (Highs[1][ck - j] >= ch) isHi = false;
            if (isHi) swHiPts.Add(new SwingPt { Bar = bar5m - ck, Price = ch });

            bool isLo = true;
            double cl = Lows[1][ck];
            for (int j = 1; j <= SwingLeftBars && isLo; j++)
                if (Lows[1][ck + j] <= cl) isLo = false;
            for (int j = 1; j <= SwingRightBars && isLo; j++)
                if (Lows[1][ck - j] <= cl) isLo = false;
            if (isLo) swLoPts.Add(new SwingPt { Bar = bar5m - ck, Price = cl });
        }

        // ================================================================
        // ZONE INVALIDATION
        // ================================================================
        private void InvalidateZones5m()
        {
            double cc = Closes[1][0];
            var surv = new List<UnifiedZone>();
            foreach (var z in activeZones)
            {
                if (z.Used || bar5m - z.BirthBar5m > MaxFvgAge) continue;
                if (z.Dir == 1 && cc < z.Bottom) continue;
                if (z.Dir == -1 && cc > z.Top) continue;
                surv.Add(z);
            }
            if (surv.Count > 50) surv = surv.Skip(surv.Count - 50).ToList();
            activeZones = surv;
        }

        // ================================================================
        // 1m BAR: Execution (entry + exit)
        // ================================================================
        private void OnBar1m()
        {
            if (CurrentBars[0] < 10 || CurrentBars[1] < 50) return;

            DateTime bc = Times[0][0];
            DateTime bo = bc.AddMinutes(-1);
            double hf = bo.Hour + bo.Minute / 60.0;

            DateTime barDate = bo.Hour >= 18 ? bo.Date.AddDays(1) : bo.Date;

            // --- Day boundary ---
            if (barDate != curDate)
            {
                foreach (var p in positions)
                {
                    double ex = p.Dir == 1 ? Closes[0][1] - slipPts : Closes[0][1] + slipPts;
                    ExitPos(p, ex, "eod_close");
                }
                positions.Clear();
                curDate = barDate;
                dayPnl = 0; consecLoss = 0; dayStopped = false;
            }

            // --- Recompute composite bias each 1m bar ---
            RecomputeCompositeBias();

            // --- Zone invalidation on 1m close (matches Python) ---
            {
                double cc1m = Closes[0][0];
                var surv = new List<UnifiedZone>();
                foreach (var z in activeZones)
                {
                    if (z.Used || bar5m - z.BirthBar5m > MaxFvgAge) continue;
                    if (z.Dir == 1 && cc1m < z.Bottom) continue;
                    if (z.Dir == -1 && cc1m > z.Top) continue;
                    surv.Add(z);
                }
                if (surv.Count > 50) surv = surv.Skip(surv.Count - 50).ToList();
                activeZones = surv;
            }

            // --- EXIT ---
            ManageExits(hf);

            // --- ENTRY ---
            if (dayStopped) return;
            if (hf < 10.0 || hf >= 16.0) return;
            if (hf >= 9.5 && hf <= 10.0) return;
            if (InNewsBlackout(bo)) return;

            TryEntry(hf);
        }

        // ================================================================
        // EXIT MANAGEMENT (1m)
        // ================================================================
        private void ManageExits(double hf)
        {
            // Phase 1: Collect exit events (Axiom 5: process losses first)
            var exitEvents = new List<(int pi, double exP, string reason, bool isLoss)>();
            for (int pi = 0; pi < positions.Count; pi++)
            {
                var p = positions[pi];
                bool exited = false; string reason = ""; double exP = 0;

                double cH = Highs[0][0], cL = Lows[0][0], cC = Closes[0][0];

                if (p.Dir == 1)
                {
                    double eff = p.Trimmed && p.TrailStop > 0 ? p.TrailStop : p.Stop;
                    if (p.Trimmed && p.BeStop > 0) eff = Math.Max(eff, p.BeStop);

                    if (cL <= eff)
                    {
                        exP = eff - slipPts;
                        reason = p.Trimmed && eff >= p.BeStop ? "be_sweep" : "stop";
                        exited = true;
                    }
                    else if (!p.Trimmed && cH >= p.TP1)
                    {
                        int tc = Math.Max(1, (int)(p.Orig * TrimPct));
                        p.Remaining = p.Orig - tc;
                        p.Trimmed = true;
                        if (UseBE) p.BeStop = p.Entry - BeOffsetR * Math.Abs(p.Entry - p.Stop);
                        if (p.Remaining > 0)
                        {
                            p.TrailStop = FindNthSwLo(NthSwing);
                            if (double.IsNaN(p.TrailStop) || p.TrailStop <= 0)
                                p.TrailStop = UseBE ? p.BeStop : p.Stop;
                        }
                        if (p.Remaining <= 0)
                        { exP = p.TP1; reason = "tp1"; exited = true; }
                        // Axiom 9: worst-case same-bar BE check after trim
                        if (!exited && WorstCaseTrimBe && UseBE && cL <= p.BeStop)
                        {
                            exP = p.BeStop - slipPts;
                            reason = "be_sweep";
                            exited = true;
                        }
                    }
                    if (p.Trimmed && !exited)
                    {
                        double nt = FindNthSwLo(NthSwing);
                        if (!double.IsNaN(nt) && nt > p.TrailStop) p.TrailStop = nt;
                    }
                }
                else // short
                {
                    double eff = p.Trimmed && p.TrailStop > 0 ? p.TrailStop : p.Stop;
                    if (p.Trimmed && p.BeStop > 0 && eff > p.BeStop) eff = p.BeStop;

                    if (cH >= eff)
                    {
                        exP = eff + slipPts;
                        reason = p.Trimmed && eff <= p.BeStop ? "be_sweep" : "stop";
                        exited = true;
                    }
                    else if (!p.Trimmed && cL <= p.TP1)
                    {
                        int tc = Math.Max(1, (int)(p.Orig * TrimPct));
                        p.Remaining = p.Orig - tc;
                        p.Trimmed = true;
                        if (UseBE) p.BeStop = p.Entry + BeOffsetR * Math.Abs(p.Entry - p.Stop);
                        if (p.Remaining > 0)
                        {
                            double ntI = FindNthSwHi(NthSwing);
                            p.TrailStop = (double.IsNaN(ntI) || ntI <= 0 || ntI > p.Entry)
                                ? (UseBE ? p.BeStop : p.Stop) : ntI;
                        }
                        if (p.Remaining <= 0)
                        { exP = p.TP1; reason = "tp1"; exited = true; }
                        // Axiom 9: worst-case same-bar BE check after trim
                        if (!exited && WorstCaseTrimBe && UseBE && cH >= p.BeStop)
                        {
                            exP = p.BeStop + slipPts;
                            reason = "be_sweep";
                            exited = true;
                        }
                    }
                    if (p.Trimmed && !exited)
                    {
                        double nt = FindNthSwHi(NthSwing);
                        if (!double.IsNaN(nt) && nt > 0 && nt < p.TrailStop) p.TrailStop = nt;
                    }
                }

                // EOD
                if (!exited && EODClose && hf >= 15.917)
                {
                    exP = p.Dir == 1 ? Closes[0][0] - slipPts : Closes[0][0] + slipPts;
                    reason = "eod_close"; exited = true;
                }

                // PA early cut (10-20 1m bars)
                if (!exited && !p.Trimmed)
                {
                    int barsIn = CurrentBars[0] - p.EntryBar1m;
                    if (barsIn >= 10 && barsIn <= 20)
                    {
                        double wickSum = 0; int favCnt = 0; int cnt = Math.Min(barsIn + 1, CurrentBars[0]);
                        for (int j = 0; j < cnt; j++)
                        {
                            double rng = Highs[0][j] - Lows[0][j];
                            double body = Math.Abs(Closes[0][j] - Opens[0][j]);
                            wickSum += rng > 0 ? 1.0 - body / rng : 0;
                            if (Math.Sign(Closes[0][j] - Opens[0][j]) == p.Dir) favCnt++;
                        }
                        double avgWick = wickSum / cnt;
                        double favorable = (double)favCnt / cnt;
                        double disp = (Closes[0][0] - p.Entry) * p.Dir;
                        double curAtr = atr5m[0];
                        if (double.IsNaN(curAtr) || curAtr <= 0) curAtr = 30;
                        if (avgWick > 0.65 && favorable < 0.5 && disp < curAtr * 0.3 && barsIn >= 15)
                        {
                            exP = Closes[0][0]; // proxy for next bar open (not available at OnBarClose)
                            reason = "early_cut_pa"; exited = true;
                        }
                    }
                }

                if (exited)
                {
                    // Estimate if this exit is a loss for sorting purposes
                    double pnlEst = (exP - p.Entry) * p.Dir;
                    exitEvents.Add((pi, exP, reason, pnlEst < 0));
                }
            }

            // Phase 2: Sort exits — losses first (Axiom 5)
            exitEvents.Sort((a, b) => b.isLoss.CompareTo(a.isLoss));

            // Phase 3: Process exits in sorted order
            var rm = new HashSet<int>();
            foreach (var ev in exitEvents)
            {
                ExitPos(positions[ev.pi], ev.exP, ev.reason);
                rm.Add(ev.pi);
            }
            for (int i = positions.Count - 1; i >= 0; i--)
                if (rm.Contains(i)) positions.RemoveAt(i);
        }

        // ================================================================
        // ENTRY (1m)
        // ================================================================
        private void TryEntry(double hf)
        {
            if (positions.Count >= MaxPositions) return;
            var occ = new HashSet<int>();
            foreach (var p in positions) occ.Add(p.Tier);

            int[] tiers = EnableChain && EnableTrend ? new[] { 1, 2 }
                : EnableChain ? new[] { 1 } : EnableTrend ? new[] { 2 } : new int[0];

            double cH = Highs[0][0], cL = Lows[0][0], cC = Closes[0][0];
            double curAtr = atr5m[0];
            if (double.IsNaN(curAtr) || curAtr <= 0) curAtr = 30;

            foreach (int tier in tiers)
            {
                if (dayStopped || occ.Contains(tier) || positions.Count >= MaxPositions) break;

                UnifiedZone best = null;
                double bEp = 0, bSp = 0, bSd = 0, bFill = double.NegativeInfinity;
                bool bSbs = false;

                foreach (var z in activeZones)
                {
                    if (z.Used || z.Tier != tier || z.BirthBar5m >= bar5m) continue;
                    double ep = z.Dir == 1 ? z.Top : z.Bottom;

                    // Fill check
                    if (z.Dir == 1 && cL > ep) continue;
                    if (z.Dir == -1 && cH < ep) continue;

                    // Zone-based stop
                    double sp = z.Dir == 1
                        ? z.Bottom - z.Size * StopBufferPct
                        : z.Top + z.Size * StopBufferPct;
                    double sd = Math.Abs(ep - sp);
                    if (TightenFactor < 1.0)
                    {
                        sp = z.Dir == 1 ? ep - sd * TightenFactor : ep + sd * TightenFactor;
                        sd = Math.Abs(ep - sp);
                    }
                    if (sd < MinStopAtrMult * curAtr || sd < 0.5) continue;

                    // SBS check
                    bool sbs = (z.Dir == 1 && cL <= sp) || (z.Dir == -1 && cH >= sp);

                    // Bias filter
                    if (z.Dir == 1 && biasDirection < 0) continue;
                    if (z.Dir == -1 && biasDirection > 0) continue;

                    // Trend PD re-check at fill time
                    if (z.Tier == 2 && onReady
                        && !double.IsNaN(onHighReady) && !double.IsNaN(onLowReady))
                    {
                        double onRng = onHighReady - onLowReady;
                        if (onRng > 0)
                        {
                            double fp = (ep - onLowReady) / onRng;
                            if (z.Dir == 1 && fp >= 0.5) continue;
                            if (z.Dir == -1 && fp < 0.5) continue;
                        }
                    }

                    double fq = -Math.Abs(cC - ep);
                    if (fq > bFill)
                    { bFill = fq; best = z; bEp = ep; bSp = sp; bSd = sd; bSbs = sbs; }
                }

                if (best == null) continue;
                best.Used = true;
                int dir = best.Dir;

                // Grade: Python _compute_grade(ba, regime)
                double ba = (dir == Math.Sign(biasDirection) && biasDirection != 0) ? 1.0 : 0.0;
                string grade = ComputeGrade(ba, regimeValue);

                bool isReduced = Times[0][0].DayOfWeek == DayOfWeek.Monday
                              || Times[0][0].DayOfWeek == DayOfWeek.Friday
                              || regimeValue < 1.0;
                double rAmt = isReduced ? ReducedR : NormalR;
                if (grade == "A+") rAmt *= APlusMult;
                else if (grade == "B+") rAmt *= BPlusMult;
                else rAmt *= 0.5;

                if (tier == 1)
                {
                    if (best.SweepRangeAtr >= BigSweepThreshold) rAmt *= BigSweepMult;
                    if (dir == -1 && hf >= 10.0 && hf < 12.0) rAmt *= AMShortMult;
                }
                else rAmt *= TrendRMult;

                int contracts = bSd > 0 ? Math.Max(1, (int)(rAmt / (bSd * pointValue))) : 0;
                if (contracts <= 0) { best.Used = false; continue; }

                // Same-bar stop (Axiom 1)
                if (bSbs)
                {
                    double exp = dir == 1 ? bSp - slipPts : bSp + slipPts;
                    double pp = (dir == 1 ? exp - bEp : bEp - exp) * pointValue * contracts;
                    pp -= CommPerSide * 2 * contracts;
                    double rr = NormalR > 0 ? pp / NormalR : 0;
                    LogTrade(Times[0][0], Times[0][0], dir, tier, grade,
                        bEp, exp, bSp, 0, "same_bar_stop", rr, pp, contracts, bSd, false);
                    dayPnl += rr; consecLoss++;
                    if (consecLoss >= MaxConsecLosses) dayStopped = true;
                    if (dayPnl <= -DailyMaxLossR) dayStopped = true;
                    continue;
                }

                double tp1 = dir == 1 ? bEp + bSd * FixedTpR : bEp - bSd * FixedTpR;
                positions.Add(new Pos {
                    Tier = tier, Dir = dir, EntryBar1m = CurrentBars[0],
                    Entry = bEp, Stop = bSp, TP1 = tp1,
                    Contracts = contracts, Orig = contracts, Remaining = contracts,
                    Trimmed = false, BeStop = 0, TrailStop = 0,
                    Grade = grade, SweepAtr = best.SweepRangeAtr,
                });
            }
        }

        // ================================================================
        // GRADE — Python: chain_engine.py _compute_grade
        // ================================================================
        private string ComputeGrade(double ba, double regime)
        {
            if (double.IsNaN(ba) || double.IsNaN(regime) || regime == 0.0) return "C";
            if (ba > 0.5 && regime >= 1.0) return "A+";
            if (ba > 0.5 || regime >= 1.0) return "B+";
            return "C";
        }

        // ================================================================
        // EXIT + PnL — Python: unified_engine_1m.py exit_position
        // ================================================================
        private void ExitPos(Pos p, double exP, string reason)
        {
            double pnlPts = (exP - p.Entry) * p.Dir;
            double totalPnl; int exC = p.Remaining;

            if (p.Trimmed && reason != "tp1")
            {
                int trimC = p.Orig - p.Remaining;
                double tPnl = (p.TP1 - p.Entry) * p.Dir * pointValue * trimC;
                totalPnl = tPnl + pnlPts * pointValue * exC;
                totalPnl -= CommPerSide * 2 * p.Orig;
                exC = p.Orig;
            }
            else
            {
                if (reason == "tp1") exC = p.Orig;
                totalPnl = pnlPts * pointValue * exC;
                totalPnl -= CommPerSide * 2 * exC;
            }

            double r = NormalR > 0 ? totalPnl / NormalR : 0;
            double sd = Math.Abs(p.Entry - p.Stop);

            LogTrade(Times[0][Math.Max(0, CurrentBars[0] - (CurrentBars[0] - p.EntryBar1m))],
                Times[0][0], p.Dir, p.Tier, p.Grade,
                p.Entry, exP, p.Stop, p.TP1, reason, r, totalPnl, p.Orig, sd, p.Trimmed);

            dayPnl += r;
            if (reason == "be_sweep" && p.Trimmed) { /* net profitable */ }
            else if (reason == "eod_close") { /* neutral for 0-for-2 */ }
            else if (r < 0) { consecLoss++; }
            else { consecLoss = 0; }

            if (consecLoss >= MaxConsecLosses) dayStopped = true;
            if (dayPnl <= -DailyMaxLossR) dayStopped = true;
        }

        // ================================================================
        // NEWS FILTER — Python: features/news_filter.py
        // ================================================================
        private void LoadNewsCalendar()
        {
            newsEvents.Clear();
            if (string.IsNullOrEmpty(NewsCsvPath) || !File.Exists(NewsCsvPath)) return;

            var lines = File.ReadAllLines(NewsCsvPath);
            var tz = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");

            for (int i = 1; i < lines.Length; i++) // skip header
            {
                var parts = lines[i].Split(',');
                if (parts.Length < 3) continue;
                string dateStr = parts[0].Trim();
                string timeStr = parts[1].Trim();
                // Parse as ET
                if (DateTime.TryParse(dateStr + " " + timeStr, out DateTime etTime))
                {
                    DateTime utcTime = TimeZoneInfo.ConvertTimeToUtc(etTime, tz);
                    newsEvents.Add(new NewsEvent
                    {
                        StartUtc = utcTime.AddMinutes(-NewsBlackoutBefore),
                        EndUtc = utcTime.AddMinutes(NewsCooldownAfter),
                    });
                }
            }
        }

        private bool InNewsBlackout(DateTime barTimeUtc)
        {
            // NinjaTrader bar time is local/exchange. Convert if needed.
            foreach (var ev in newsEvents)
            {
                if (barTimeUtc >= ev.StartUtc && barTimeUtc <= ev.EndUtc)
                    return true;
            }
            return false;
        }

        // ================================================================
        // TRADE LOG
        // ================================================================
        private void LogTrade(DateTime et, DateTime xt, int dir, int tier, string grade,
            double ep, double xp, double sp, double tp, string reason,
            double r, double pnl, int contracts, double sd, bool trimmed)
        {
            tradeLog.AppendLine(
                $"{et:yyyy-MM-dd HH:mm},{xt:yyyy-MM-dd HH:mm},{dir},{tier},{grade}," +
                $"{ep:F2},{xp:F2},{sp:F2},{tp:F2},{reason},{r:F4},{pnl:F2}," +
                $"{contracts},{sd:F2},{trimmed}");
            tradeCount++; totalR += r;
            if (r > 0) totWins += r; else totLosses += Math.Abs(r);
        }

        // ================================================================
        // SWING HELPERS
        // ================================================================
        private double FindNthSwLo(int n)
        {
            return swLoPts.Count >= n ? swLoPts[swLoPts.Count - n].Price : double.NaN;
        }
        private double FindNthSwHi(int n)
        {
            return swHiPts.Count >= n ? swHiPts[swHiPts.Count - n].Price : double.NaN;
        }
    }
}
