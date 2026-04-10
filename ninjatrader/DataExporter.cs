// DataExporter.cs — Export NT8 1m bar data to CSV for Python comparison
// Run on MNQ in Strategy Analyzer to dump raw bar timestamps + OHLCV
// This tells us EXACTLY what NT8 sees (timezone, prices, bar count)

#region Using declarations
using System;
using System.IO;
using System.Text;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class DataExporter : Strategy
    {
        private StringBuilder sb;
        private int barCount;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Export 1m bar data to CSV for Python comparison";
                Name = "DataExporter";
                Calculate = Calculate.OnBarClose;
                BarsRequiredToTrade = 0;
            }
            else if (State == State.DataLoaded)
            {
                sb = new StringBuilder();
                sb.AppendLine("bar_close_time,open,high,low,close,volume,bar_index");
                barCount = 0;
            }
            else if (State == State.Terminated)
            {
                if (sb != null && barCount > 0)
                {
                    string path = Path.Combine(
                        Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                        "NT8_MNQ_1m_export.csv");
                    File.WriteAllText(path, sb.ToString());
                    Print($"DataExporter: {barCount} bars exported to {path}");
                }
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0) return;

            sb.AppendLine($"{Time[0]:yyyy-MM-dd HH:mm:ss},{Open[0]},{High[0]},{Low[0]},{Close[0]},{Volume[0]},{CurrentBar}");
            barCount++;

            // Print first bar for timezone verification
            if (barCount == 1)
                Print($"[EXPORT] First bar: Time={Time[0]:yyyy-MM-dd HH:mm:ss}, O={Open[0]}, C={Close[0]}");
            // Print periodic status
            if (barCount % 100000 == 0)
                Print($"[EXPORT] {barCount} bars processed...");
        }
    }
}
