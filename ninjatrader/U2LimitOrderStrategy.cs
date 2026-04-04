// =============================================================================
// U2LimitOrderStrategy.cs — NinjaTrader 8 NinjaScript
// FVG Zone Limit Order Strategy (Long-Only)
//
// Port of experiments/u2_clean.py (audited, 4 fixes applied)
// Config: A2 sz>0.3 age<200 min>5pt tf=0.80
//
// Expected: ~2012t, PF=1.87, R=+1270, PPDD=48.4, MaxDD=26.3R, 0 neg yrs
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
    public class U2LimitOrderStrategy : Strategy
    {
        // ================================================================
        // Parameters (match u2_clean.py best config)
        // ================================================================
        [NinjaScriptProperty] [Display(Name="FVG Size ATR Mult", Order=1, GroupName="FVG Zone")]
        public double FvgSizeMult { get; set; }

        [NinjaScriptProperty] [Display(Name="Max FVG Age (bars)", Order=2, GroupName="FVG Zone")]
        public int MaxFvgAge { get; set; }

        [NinjaScriptProperty] [Display(Name="Stop Buffer Pct (A2)", Order=3, GroupName="Stop")]
        public double StopBufferPct { get; set; }

        [NinjaScriptProperty] [Display(Name="Stop Tighten Factor", Order=4, GroupName="Stop")]
        public double TightenFactor { get; set; }

        [NinjaScriptProperty] [Display(Name="Min Stop Points", Order=5, GroupName="Stop")]
        public double MinStopPts { get; set; }

        [NinjaScriptProperty] [Display(Name="Trim Pct at TP1", Order=6, GroupName="Exit")]
        public double TrimPct { get; set; }

        [NinjaScriptProperty] [Display(Name="TP IRL Multiplier", Order=7, GroupName="Exit")]
        public double TpMult { get; set; }

        [NinjaScriptProperty] [Display(Name="Nth Swing Trail", Order=8, GroupName="Exit")]
        public int NthSwing { get; set; }

        [NinjaScriptProperty] [Display(Name="Risk Per Trade ($)", Order=9, GroupName="Risk")]
        public double RiskDollars { get; set; }

        [NinjaScriptProperty] [Display(Name="Reduced Risk ($)", Order=10, GroupName="Risk")]
        public double ReducedRisk { get; set; }

        [NinjaScriptProperty] [Display(Name="Daily Max Loss (R)", Order=11, GroupName="Risk")]
        public double DailyMaxLossR { get; set; }

        [NinjaScriptProperty] [Display(Name="Max Consecutive Losses", Order=12, GroupName="Risk")]
        public int MaxConsecLosses { get; set; }

        [NinjaScriptProperty] [Display(Name="Slippage Ticks (stop exit)", Order=13, GroupName="Execution")]
        public int SlippageTicks { get; set; }

        [NinjaScriptProperty] [Display(Name="Commission Per Side", Order=14, GroupName="Execution")]
        public double CommissionPerSide { get; set; }

        [NinjaScriptProperty] [Display(Name="Swing Left Bars", Order=15, GroupName="Swing")]
        public int SwingLeftBars { get; set; }

        [NinjaScriptProperty] [Display(Name="Swing Right Bars", Order=16, GroupName="Swing")]
        public int SwingRightBars { get; set; }

        [NinjaScriptProperty] [Display(Name="EOD Close", Order=17, GroupName="Session")]
        public bool EODClose { get; set; }

        [NinjaScriptProperty] [Display(Name="PA Early Cut", Order=18, GroupName="Exit")]
        public bool PAEarlyCut { get; set; }

        // ================================================================
        // Internal state
        // ================================================================
        private ATR atrIndicator;
        private double pointValue;

        // FVG zone tracking
        private class FvgZone
        {
            public bool IsBull;
            public double Top;
            public double Bottom;
            public double Size;
            public int BirthBar;
            public double BirthAtr;
            public bool Used;
        }
        private List<FvgZone> activeZones;

        // Swing tracking
        private List<double> swingHighPrices;
        private List<int>    swingHighBars;
        private List<double> swingLowPrices;
        private List<int>    swingLowBars;

        // Position state
        private class TradeState
        {
            public int Direction;          // +1 long only
            public double EntryPrice;
            public double StopPrice;
            public double TP1Price;
            public int Contracts;
            public int OrigContracts;
            public int EntryBarIdx;
            public DateTime EntryTime;
            public bool Trimmed;
            public double BeStop;
            public double TrailStop;
            public double OrigStopDist;
        }
        private TradeState ts;

        // Daily state
        private DateTime currentDate;
        private double dailyPnlR;
        private int consecLosses;
        private bool dayStopped;

        // Bar counter
        private int barIdx;

        // Pending early cut
        private bool pendingEarlyCut;

        // FVG detection state (3-candle pattern)
        // After shift(1): we store previous bar's detection and emit it 1 bar later
        private bool pendingBullFvg;
        private double pendingBullTop, pendingBullBottom;
        private bool pendingBearFvg;
        private double pendingBearTop, pendingBearBottom;

        // Trade log for CSV export
        private StringBuilder tradeLog;
        private int tradeCount;
        private double totalR;

        // ================================================================
        // Initialization
        // ================================================================
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "U2 Limit Order at FVG Zone (Long-Only, Audited)";
                Name = "U2LimitOrder";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = false;
                MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                StartBehavior = StartBehavior.WaitUntilFlat;
                BarsRequiredToTrade = 50;

                // Defaults (best config from u2_clean.py)
                FvgSizeMult     = 0.3;
                MaxFvgAge       = 200;
                StopBufferPct   = 0.15;
                TightenFactor   = 0.80;
                MinStopPts      = 5.0;
                TrimPct         = 0.25;
                TpMult          = 2.0;
                NthSwing        = 3;
                RiskDollars     = 1000;
                ReducedRisk     = 500;
                DailyMaxLossR   = 2.0;
                MaxConsecLosses = 2;
                SlippageTicks   = 1;
                CommissionPerSide = 0.62;
                SwingLeftBars   = 3;
                SwingRightBars  = 1;
                EODClose        = true;
                PAEarlyCut      = true;
            }
            else if (State == State.Configure)
            {
                // Primary instrument is NQ/MNQ 5-minute (auto-added)
                // No additional data series needed for U2
            }
            else if (State == State.DataLoaded)
            {
                atrIndicator = ATR(14);
                pointValue = Instrument.MasterInstrument.PointValue;

                activeZones    = new List<FvgZone>();
                swingHighPrices = new List<double>();
                swingHighBars   = new List<int>();
                swingLowPrices  = new List<double>();
                swingLowBars    = new List<int>();

                ts = null;
                currentDate = DateTime.MinValue;
                dailyPnlR = 0;
                consecLosses = 0;
                dayStopped = false;
                barIdx = 0;
                pendingEarlyCut = false;
                pendingBullFvg = false;
                pendingBearFvg = false;

                tradeLog = new StringBuilder();
                tradeLog.AppendLine("entry_time,exit_time,dir,entry_price,exit_price,stop_price,tp1_price,reason,r,pnl_dollars,contracts,stop_dist");
                tradeCount = 0;
                totalR = 0;
            }
            else if (State == State.Terminated)
            {
                if (tradeLog != null && tradeCount > 0)
                {
                    string path = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                        "U2_NT8_trades.csv");
                    File.WriteAllText(path, tradeLog.ToString());
                    Print($"U2 FINAL: {tradeCount} trades, R={totalR:+0.0}. CSV: {path}");
                }
            }
        }

        // ================================================================
        // Main bar update
        // ================================================================
        protected override void OnBarUpdate()
        {
            if (CurrentBar < 50) return;
            barIdx++;

            // Bar time in ET (NinjaTrader Time[0] = bar CLOSE time)
            DateTime barCloseET = Time[0];
            DateTime barOpenET  = barCloseET.AddMinutes(-BarsPeriod.Value);
            double hf = barOpenET.Hour + barOpenET.Minute / 60.0;

            double curAtr = atrIndicator[0];
            if (double.IsNaN(curAtr) || curAtr <= 0) curAtr = 30.0;

            // ---- DAILY RESET ----
            DateTime barDate = barOpenET.Hour >= 18
                ? barOpenET.Date.AddDays(1)
                : barOpenET.Date;

            if (barDate != currentDate)
            {
                currentDate = barDate;
                dailyPnlR = 0;
                consecLosses = 0;
                dayStopped = false;
                pendingEarlyCut = false;  // reset across day boundary
            }

            // ---- PHASE A: REGISTER NEW FVG ZONES (shifted by 1 bar) ----
            // Emit PREVIOUS bar's pending FVG (shift(1) anti-lookahead)
            if (pendingBullFvg)
            {
                double zoneSize = pendingBullTop - pendingBullBottom;
                if (zoneSize >= FvgSizeMult * curAtr && zoneSize > 0)
                {
                    activeZones.Add(new FvgZone
                    {
                        IsBull = true,
                        Top = pendingBullTop,
                        Bottom = pendingBullBottom,
                        Size = zoneSize,
                        BirthBar = barIdx,
                        BirthAtr = curAtr,
                        Used = false,
                    });
                }
                pendingBullFvg = false;
            }
            if (pendingBearFvg)
            {
                double zoneSize = pendingBearTop - pendingBearBottom;
                if (zoneSize >= FvgSizeMult * curAtr && zoneSize > 0)
                {
                    activeZones.Add(new FvgZone
                    {
                        IsBull = false,
                        Top = pendingBearTop,
                        Bottom = pendingBearBottom,
                        Size = zoneSize,
                        BirthBar = barIdx,
                        BirthAtr = curAtr,
                        Used = false,
                    });
                }
                pendingBearFvg = false;
            }

            // Detect NEW FVGs on THIS bar (will be emitted NEXT bar)
            // FIX #4: Skip if any of the 3 candles is on a rollover date
            if (CurrentBar >= 2)
            {
                bool isRoll = IsRolloverBar(0) || IsRolloverBar(1) || IsRolloverBar(2);
                if (!isRoll)
                {
                    double h1 = High[2], l1 = Low[2];  // candle-1
                    double h3 = High[0], l3 = Low[0];  // candle-3 (current)

                    // Bullish FVG: candle1.high < candle3.low
                    if (h1 < l3)
                    {
                        pendingBullFvg = true;
                        pendingBullTop = l3;       // gap top = candle3.low
                        pendingBullBottom = h1;    // gap bottom = candle1.high
                    }

                    // Bearish FVG: candle1.low > candle3.high
                    if (l1 > h3)
                    {
                        pendingBearFvg = true;
                        pendingBearTop = l1;       // gap top = candle1.low
                        pendingBearBottom = h3;    // gap bottom = candle3.high
                    }
                }
            }

            // ---- PHASE B: INVALIDATE / EXPIRE ZONES ----
            var surviving = new List<FvgZone>();
            foreach (var zone in activeZones)
            {
                if (zone.Used) continue;
                if ((barIdx - zone.BirthBar) > MaxFvgAge) continue;
                if (zone.IsBull && Close[0] < zone.Bottom) continue;   // invalidated
                if (!zone.IsBull && Close[0] > zone.Top) continue;     // invalidated
                surviving.Add(zone);
            }
            if (surviving.Count > 30)
                surviving = surviving.Skip(surviving.Count - 30).ToList();
            activeZones = surviving;

            // ---- UPDATE SWINGS ----
            UpdateSwings();

            // ---- PHASE C: EXIT MANAGEMENT ----
            if (pendingEarlyCut && ts != null)
            {
                ExitTrade(Open[0], "early_cut_pa");
                pendingEarlyCut = false;
            }

            if (ts != null)
            {
                ManagePosition(hf, curAtr);
            }

            // ---- PHASE D: ENTRY (limit order fill) ----
            if (ts != null) return;
            if (dayStopped) return;

            // Session filter: NY only, after 10:00 ET, before 16:00 ET
            if (hf >= 9.5 && hf <= 10.0) return;  // observation window
            if (hf < 10.0 || hf >= 16.0) return;   // NY only

            // Long-only: block all shorts
            // Find best fill among active zones
            FvgZone bestZone = null;
            double bestEntry = 0, bestStop = 0, bestStopDist = 0, bestFillQuality = double.NegativeInfinity;

            foreach (var zone in activeZones)
            {
                if (zone.Used || zone.BirthBar >= barIdx) continue;

                // LONG ONLY: skip bear zones
                if (!zone.IsBull) continue;

                double entryP = zone.Top;
                double stopP  = zone.Bottom - zone.Size * StopBufferPct;  // A2 strategy

                double stopDist = Math.Abs(entryP - stopP);

                // Apply tightening BEFORE fill-bar check (AUDIT FIX #2)
                if (TightenFactor < 1.0)
                {
                    stopP = entryP - stopDist * TightenFactor;
                    stopDist = Math.Abs(entryP - stopP);
                }

                // Min stop filters
                if (stopDist < MinStopPts) continue;
                if (stopDist < 1.0) continue;

                // Fill check: bar low must reach entry, but NOT breach tightened stop
                if (Low[0] > entryP) continue;      // price didn't reach limit
                if (Low[0] <= stopP) continue;       // stop also hit, skip

                // Bias alignment (simplified: use price vs SMA as proxy)
                // In Python we use bias_dir_arr from cache. In NT8 we use simple trend proxy.
                // For validation purposes, we skip bias filter to match raw signal count.
                // TODO: implement proper bias from 1H/4H if needed for deployment.

                // Quality: prefer zone closest to current price
                double fillQuality = -Math.Abs(Close[0] - entryP);
                if (fillQuality > bestFillQuality)
                {
                    bestFillQuality = fillQuality;
                    bestZone = zone;
                    bestEntry = entryP;
                    bestStop = stopP;
                    bestStopDist = stopDist;
                }
            }

            if (bestZone == null) return;

            // Lunch dead zone (12:30-13:00 ET) — check BEFORE marking zone used
            if (hf >= 12.5 && hf < 13.0) return;  // FIX #1: don't consume zone

            bestZone.Used = true;

            // Grade + sizing (FIX #2: match Python grade system)
            // Simplified grade: without bias/regime cache, use flat B+ grade
            // For proper NT8 deployment, add 1H/4H data series for bias/regime
            double rAmount = RiskDollars;
            bool isReduced = (barOpenET.DayOfWeek == DayOfWeek.Monday
                           || barOpenET.DayOfWeek == DayOfWeek.Friday);
            // FIX #3: regime check would go here if 1H/4H available
            if (isReduced) rAmount = ReducedRisk;

            // Grade multiplier (default B+ = 1.0x without regime data)
            // Python: A+=1.5x, B+=1.0x, C=0.5x
            // Without regime/bias, we can't compute grade. Use 1.0x as conservative default.
            double gradeMult = 1.0;
            rAmount *= gradeMult;

            int contracts = Math.Max(1, (int)(rAmount / (bestStopDist * pointValue)));
            if (contracts <= 0)
            {
                bestZone.Used = false;  // FIX #1b: release zone if can't trade
                return;
            }

            // TP computation: IRL target (most recent swing high) * tp_mult
            // Python uses irl_high_arr[i] = shift(1).ffill() of swing_high_price
            // This is simply the MOST RECENT confirmed swing high price,
            // regardless of whether it's above or below entry.
            // If it's <= entry, fallback to entry + 2*stop_dist.
            double irlTarget = FindMostRecentSwingHigh();
            if (double.IsNaN(irlTarget) || irlTarget <= bestEntry)
                irlTarget = bestEntry + bestStopDist * 2.0;
            double tpDist = irlTarget - bestEntry;
            double tp1Price = bestEntry + tpDist * TpMult;

            // Enter position (no slippage on limit order)
            EnterLong(contracts, "U2Long");

            ts = new TradeState
            {
                Direction     = 1,
                EntryPrice    = bestEntry,
                StopPrice     = bestStop,
                TP1Price      = tp1Price,
                Contracts     = contracts,
                OrigContracts = contracts,
                EntryBarIdx   = barIdx,
                EntryTime     = barCloseET,
                Trimmed       = false,
                BeStop        = 0,
                TrailStop     = 0,
                OrigStopDist  = bestStopDist,
            };
        }

        // ================================================================
        // Position management
        // ================================================================
        private void ManagePosition(double hf, double curAtr)
        {
            bool exited = false;
            string exitReason = "";
            double exitPrice = 0;
            int exitContracts = ts.Contracts;
            double slippagePts = SlippageTicks * 0.25;

            int barsInTrade = barIdx - ts.EntryBarIdx;

            // AUDIT FIX #4: Stop/TP checked BEFORE EOD close
            // --- LONG STOP / TP ---
            double effStop = ts.Trimmed && ts.TrailStop > 0 ? ts.TrailStop : ts.StopPrice;
            if (ts.Trimmed && ts.BeStop > 0)
                effStop = Math.Max(effStop, ts.BeStop);

            if (Low[0] <= effStop)
            {
                exitPrice = effStop - slippagePts;  // AUDIT FIX #3: slippage on stop
                exitReason = (ts.Trimmed && effStop >= ts.EntryPrice) ? "be_sweep" : "stop";
                exited = true;
            }
            else if (!ts.Trimmed && High[0] >= ts.TP1Price)
            {
                int trimC = Math.Max(1, (int)(ts.OrigContracts * TrimPct));
                ts.Contracts = ts.OrigContracts - trimC;
                ts.Trimmed = true;
                ts.BeStop = ts.EntryPrice;

                if (ts.Contracts > 0)
                {
                    ts.TrailStop = FindNthSwingLow(NthSwing);
                    if (double.IsNaN(ts.TrailStop) || ts.TrailStop <= 0)
                        ts.TrailStop = ts.BeStop;

                    // Partial exit
                    ExitLong(trimC, "TrimU2", "U2Long");
                }

                if (ts.Contracts <= 0)
                {
                    exitPrice = ts.TP1Price;
                    exitReason = "tp1";
                    exitContracts = ts.OrigContracts;
                    exited = true;
                }
            }

            // Trail update
            if (ts != null && ts.Trimmed && !exited)
            {
                double nt = FindNthSwingLow(NthSwing);
                if (!double.IsNaN(nt) && nt > ts.TrailStop)
                    ts.TrailStop = nt;
            }

            // --- EOD CLOSE (AFTER stop/TP — AUDIT FIX #4) ---
            if (!exited && EODClose && hf >= 15.917)
            {
                exitPrice = Close[0] - 0.25;  // 1 tick slippage
                exitReason = "eod_close";
                exited = true;
            }

            // --- PA EARLY CUT (deferred to next bar open) ---
            // Python: sets exited=True with exit_price=o[i+1] on SAME bar.
            // C#: must defer because we don't have next bar's open yet.
            // Use pendingEarlyCut flag, execute at next bar's Open[0].
            if (!exited && PAEarlyCut && !ts.Trimmed && barsInTrade >= 3 && barsInTrade <= 4)
            {
                int count = barsInTrade + 1;
                if (count > CurrentBar + 1) count = CurrentBar + 1;
                double wickSum = 0;
                int favCount = 0;

                for (int j = 0; j < count; j++)
                {
                    double rng = High[j] - Low[j];
                    double body = Math.Abs(Close[j] - Open[j]);
                    wickSum += rng > 0 ? 1.0 - body / rng : 0;
                    if (Math.Sign(Close[j] - Open[j]) == ts.Direction) favCount++;
                }
                double avgWick = wickSum / count;
                double favorable = (double)favCount / count;

                double disp = Close[0] - ts.EntryPrice;
                if (avgWick > 0.65 && favorable < 0.5 && disp < curAtr * 0.3)
                {
                    pendingEarlyCut = true;
                    return;
                }
            }

            if (exited)
            {
                ExitTrade(exitPrice, exitReason);
            }
        }

        // ================================================================
        // Exit trade and log
        // ================================================================
        private void ExitTrade(double exitPrice, string reason)
        {
            if (ts == null) return;

            // Close remaining position
            if (ts.Contracts > 0)
                ExitLong(ts.Contracts, "Exit_" + reason, "U2Long");

            // PnL calculation (CORRECT single-TP)
            double pnlPts = exitPrice - ts.EntryPrice;
            double totalPnl;

            if (ts.Trimmed && reason != "tp1")
            {
                double trimPnl = ts.TP1Price - ts.EntryPrice;
                int trimC = ts.OrigContracts - ts.Contracts;
                totalPnl = trimPnl * pointValue * trimC + pnlPts * pointValue * ts.Contracts;
                totalPnl -= CommissionPerSide * 2 * ts.OrigContracts;
            }
            else
            {
                int exitC = reason == "tp1" ? ts.OrigContracts : ts.Contracts;
                totalPnl = pnlPts * pointValue * exitC;
                totalPnl -= CommissionPerSide * 2 * exitC;
            }

            double totalRisk = ts.OrigStopDist * pointValue * ts.OrigContracts;
            double rMult = totalRisk > 0 ? totalPnl / totalRisk : 0;

            // Log
            tradeLog.AppendLine($"{ts.EntryTime:yyyy-MM-dd HH:mm},{Time[0]:yyyy-MM-dd HH:mm},1," +
                $"{ts.EntryPrice:F2},{exitPrice:F2},{ts.StopPrice:F2},{ts.TP1Price:F2}," +
                $"{reason},{rMult:F4},{totalPnl:F2},{ts.OrigContracts},{ts.OrigStopDist:F2}");
            tradeCount++;
            totalR += rMult;

            // Daily risk management
            dailyPnlR += rMult;

            if (reason == "be_sweep" && ts.Trimmed)
            {
                // BE sweep after trim = net profitable, don't count as loss
            }
            else if (reason == "eod_close")
            {
                // EOD close neutral for 0-for-2
            }
            else if (rMult < 0)
            {
                consecLosses++;
            }
            else
            {
                consecLosses = 0;
            }

            if (consecLosses >= MaxConsecLosses) dayStopped = true;
            if (dailyPnlR <= -DailyMaxLossR) dayStopped = true;

            ts = null;
        }

        // ================================================================
        // Swing detection (fractal)
        // ================================================================
        private void UpdateSwings()
        {
            if (CurrentBar < SwingLeftBars + SwingRightBars + 1) return;

            int checkBar = SwingRightBars;  // check the bar that is RightBars ago

            // Swing High: high[checkBar] > all lefts AND all rights
            bool isSwingHigh = true;
            double candHigh = High[checkBar];
            for (int j = 1; j <= SwingLeftBars; j++)
            {
                if (High[checkBar + j] >= candHigh) { isSwingHigh = false; break; }
            }
            if (isSwingHigh)
            {
                for (int j = 1; j <= SwingRightBars; j++)
                {
                    if (High[checkBar - j] >= candHigh) { isSwingHigh = false; break; }
                }
            }
            if (isSwingHigh)
            {
                swingHighPrices.Add(candHigh);
                swingHighBars.Add(barIdx - checkBar);
            }

            // Swing Low
            bool isSwingLow = true;
            double candLow = Low[checkBar];
            for (int j = 1; j <= SwingLeftBars; j++)
            {
                if (Low[checkBar + j] <= candLow) { isSwingLow = false; break; }
            }
            if (isSwingLow)
            {
                for (int j = 1; j <= SwingRightBars; j++)
                {
                    if (Low[checkBar - j] <= candLow) { isSwingLow = false; break; }
                }
            }
            if (isSwingLow)
            {
                swingLowPrices.Add(candLow);
                swingLowBars.Add(barIdx - checkBar);
            }
        }

        private double FindNthSwingLow(int n)
        {
            if (swingLowPrices.Count < n) return double.NaN;
            return swingLowPrices[swingLowPrices.Count - n];
        }

        private double FindNthSwingHigh(int n)
        {
            if (swingHighPrices.Count < n) return double.NaN;
            return swingHighPrices[swingHighPrices.Count - n];
        }

        private double FindMostRecentSwingHigh()
        {
            if (swingHighPrices.Count == 0) return double.NaN;
            return swingHighPrices[swingHighPrices.Count - 1];
        }

        private double FindNearestSwingHigh(double abovePrice)
        {
            for (int i = swingHighPrices.Count - 1; i >= 0; i--)
            {
                if (swingHighPrices[i] > abovePrice)
                    return swingHighPrices[i];
            }
            return double.NaN;
        }

        // FIX #4: Rollover date detection
        // NQ quarterly rolls: 2nd Friday of March, June, September, December
        // Check if bar at barsAgo is on a roll date
        private bool IsRolloverBar(int barsAgo)
        {
            if (CurrentBar < barsAgo) return false;
            DateTime barTime = Time[barsAgo];
            // Roll dates: 2nd Thursday of expiry month (rollover typically happens day before expiry)
            // Simplified: check if it's the 2nd week of Mar/Jun/Sep/Dec
            int month = barTime.Month;
            if (month != 3 && month != 6 && month != 9 && month != 12)
                return false;
            int day = barTime.Day;
            // 2nd Friday is typically day 8-14. Roll activity spans Thu-Fri.
            return day >= 8 && day <= 15;
        }
    }
}
